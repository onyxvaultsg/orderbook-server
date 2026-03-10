"""
Onyx Vault — Orderbook Server with Supabase
Analyze. Acquire. Profit.

Persistent orderbook with user authentication.
Orders and trades survive server restarts via Supabase PostgreSQL.

Environment variables required on Render:
    SUPABASE_URL=https://xxxxx.supabase.co
    SUPABASE_SERVICE_KEY=eyJhbG...  (service_role key, NOT anon key)
"""

from __future__ import annotations

import os
import json
import uuid
import time
import asyncio
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from supabase import create_client, Client


# ── Supabase Client ─────────────────────────────────────────────

SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_KEY", "")

db: Optional[Client] = None

def get_db() -> Optional[Client]:
    global db
    if db is None and SUPABASE_URL and SUPABASE_KEY:
        db = create_client(SUPABASE_URL, SUPABASE_KEY)
    return db


# ── Enums & Data Models ─────────────────────────────────────────

class Side(str, Enum):
    BID = "BID"
    ASK = "ASK"


class OrderStatus(str, Enum):
    OPEN = "OPEN"
    PARTIAL = "PARTIAL"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"


@dataclass
class Order:
    side: Side
    price: float
    qty: int
    trader: str
    trader_id: Optional[str] = None
    order_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    timestamp: float = field(default_factory=time.time)
    filled_qty: int = 0
    status: OrderStatus = OrderStatus.OPEN

    @property
    def remaining(self) -> int:
        return self.qty - self.filled_qty

    def fill(self, fill_qty: int) -> None:
        self.filled_qty += fill_qty
        self.status = OrderStatus.FILLED if self.remaining == 0 else OrderStatus.PARTIAL

    def cancel(self) -> None:
        self.status = OrderStatus.CANCELLED

    def to_dict(self) -> dict:
        return {
            "order_id": self.order_id,
            "side": self.side.value,
            "price": self.price,
            "qty": self.qty,
            "filled_qty": self.filled_qty,
            "remaining": self.remaining,
            "trader": self.trader,
            "trader_id": self.trader_id,
            "status": self.status.value,
            "timestamp": self.timestamp,
        }


@dataclass
class Trade:
    trade_id: str
    card_id: str
    price: float
    qty: int
    buyer: str
    seller: str
    buyer_id: Optional[str]
    seller_id: Optional[str]
    buyer_fee: float
    seller_fee: float
    platform_revenue: float
    timestamp: float

    def to_dict(self) -> dict:
        return {
            "trade_id": self.trade_id,
            "card_id": self.card_id,
            "price": self.price,
            "qty": self.qty,
            "buyer": self.buyer,
            "seller": self.seller,
            "buyer_fee": round(self.buyer_fee, 4),
            "seller_fee": round(self.seller_fee, 4),
            "platform_revenue": round(self.platform_revenue, 4),
            "timestamp": self.timestamp,
        }


# ── Orderbook Engine ────────────────────────────────────────────

from sortedcontainers import SortedList

class OrderBook:
    def __init__(self, card_id: str, fee_bps: int = 150):
        self.card_id = card_id
        self.fee_bps = fee_bps
        self._bids: SortedList[Order] = SortedList(key=lambda o: (-o.price, o.timestamp))
        self._asks: SortedList[Order] = SortedList(key=lambda o: (o.price, o.timestamp))
        self._orders: dict[str, Order] = {}
        self.trades: list[Trade] = []
        self.total_platform_revenue: float = 0.0

    def submit_order(self, order: Order) -> list[Trade]:
        new_trades = self._match(order)
        if order.remaining > 0 and order.status != OrderStatus.CANCELLED:
            book_side = self._bids if order.side == Side.BID else self._asks
            book_side.add(order)
            self._orders[order.order_id] = order
        return new_trades

    def cancel_order(self, order_id: str) -> Optional[Order]:
        order = self._orders.pop(order_id, None)
        if order is None:
            return None
        book_side = self._bids if order.side == Side.BID else self._asks
        try:
            book_side.remove(order)
        except ValueError:
            pass
        order.cancel()
        return order

    def _match(self, aggressor: Order) -> list[Trade]:
        trades: list[Trade] = []
        opposite = self._asks if aggressor.side == Side.BID else self._bids

        while aggressor.remaining > 0 and len(opposite) > 0:
            best = opposite[0]
            if aggressor.side == Side.BID and aggressor.price < best.price:
                break
            if aggressor.side == Side.ASK and aggressor.price > best.price:
                break

            fill_qty = min(aggressor.remaining, best.remaining)
            fill_price = best.price
            notional = fill_price * fill_qty
            fee_per_side = notional * (self.fee_bps / 10_000)
            platform_rev = fee_per_side * 2

            if aggressor.side == Side.BID:
                buyer, seller = aggressor.trader, best.trader
                buyer_id, seller_id = aggressor.trader_id, best.trader_id
            else:
                buyer, seller = best.trader, aggressor.trader
                buyer_id, seller_id = best.trader_id, aggressor.trader_id

            trade = Trade(
                trade_id=uuid.uuid4().hex[:8],
                card_id=self.card_id,
                price=fill_price,
                qty=fill_qty,
                buyer=buyer,
                seller=seller,
                buyer_id=buyer_id,
                seller_id=seller_id,
                buyer_fee=fee_per_side,
                seller_fee=fee_per_side,
                platform_revenue=platform_rev,
                timestamp=time.time(),
            )
            trades.append(trade)
            self.trades.append(trade)
            self.total_platform_revenue += platform_rev

            aggressor.fill(fill_qty)
            best.fill(fill_qty)

            if best.remaining == 0:
                opposite.remove(best)
                self._orders.pop(best.order_id, None)

        return trades

    @property
    def best_bid(self) -> Optional[float]:
        return self._bids[0].price if self._bids else None

    @property
    def best_ask(self) -> Optional[float]:
        return self._asks[0].price if self._asks else None

    def depth(self, levels: int = 8) -> dict:
        bid_levels: dict[float, int] = {}
        for o in self._bids:
            bid_levels[o.price] = bid_levels.get(o.price, 0) + o.remaining
            if len(bid_levels) >= levels:
                break
        ask_levels: dict[float, int] = {}
        for o in self._asks:
            ask_levels[o.price] = ask_levels.get(o.price, 0) + o.remaining
            if len(ask_levels) >= levels:
                break
        return {
            "bids": [{"price": p, "qty": q} for p, q in bid_levels.items()],
            "asks": [{"price": p, "qty": q} for p, q in ask_levels.items()],
        }

    def snapshot(self) -> dict:
        bb = self.best_bid
        ba = self.best_ask
        return {
            "card_id": self.card_id,
            "fee_bps": self.fee_bps,
            "best_bid": bb,
            "best_ask": ba,
            "spread": round(ba - bb, 2) if bb and ba else None,
            "mid_price": round((bb + ba) / 2, 2) if bb and ba else None,
            "bid_orders": len(self._bids),
            "ask_orders": len(self._asks),
            "total_trades": len(self.trades),
            "total_platform_revenue": round(self.total_platform_revenue, 2),
            "depth": self.depth(),
            "recent_trades": [t.to_dict() for t in self.trades[-20:]],
        }


# ── Card Definitions ────────────────────────────────────────────

CARDS = [
    {"id": "SV01-025-PSA10", "name": "Charizard ex SV"},
    {"id": "SWSH12-GG70-PSA10", "name": "Giratina V Alt Art"},
    {"id": "SM12-SV49-PSA10", "name": "Charizard VMAX"},
]


# ── Database Persistence ────────────────────────────────────────

def save_order_to_db(order: Order, card_id: str):
    """Save or update an order in Supabase."""
    client = get_db()
    if not client:
        return
    try:
        client.table("orders").upsert({
            "id": order.order_id,
            "card_id": card_id,
            "side": order.side.value,
            "price": float(order.price),
            "qty": order.qty,
            "filled_qty": order.filled_qty,
            "trader_id": order.trader_id,
            "trader_name": order.trader,
            "status": order.status.value,
        }).execute()
    except Exception as e:
        print(f"DB save order error: {e}")


def save_trade_to_db(trade: Trade):
    """Save a trade to Supabase."""
    client = get_db()
    if not client:
        return
    try:
        client.table("trades").insert({
            "id": trade.trade_id,
            "card_id": trade.card_id,
            "price": float(trade.price),
            "qty": trade.qty,
            "buyer_id": trade.buyer_id,
            "seller_id": trade.seller_id,
            "buyer_name": trade.buyer,
            "seller_name": trade.seller,
            "buyer_fee": float(trade.buyer_fee),
            "seller_fee": float(trade.seller_fee),
            "platform_revenue": float(trade.platform_revenue),
        }).execute()
    except Exception as e:
        print(f"DB save trade error: {e}")


def load_orders_from_db(card_id: str) -> list[Order]:
    """Load open/partial orders from Supabase on startup."""
    client = get_db()
    if not client:
        return []
    try:
        result = client.table("orders").select("*").eq(
            "card_id", card_id
        ).in_("status", ["OPEN", "PARTIAL"]).execute()

        orders = []
        for row in result.data:
            orders.append(Order(
                side=Side(row["side"]),
                price=float(row["price"]),
                qty=row["qty"],
                trader=row["trader_name"],
                trader_id=row.get("trader_id"),
                order_id=row["id"],
                filled_qty=row["filled_qty"],
                status=OrderStatus(row["status"]),
            ))
        return orders
    except Exception as e:
        print(f"DB load orders error: {e}")
        return []


def load_trades_from_db(card_id: str, limit: int = 50) -> list[Trade]:
    """Load recent trades from Supabase on startup."""
    client = get_db()
    if not client:
        return []
    try:
        result = client.table("trades").select("*").eq(
            "card_id", card_id
        ).order("created_at", desc=True).limit(limit).execute()

        trades = []
        for row in result.data:
            trades.append(Trade(
                trade_id=row["id"],
                card_id=row["card_id"],
                price=float(row["price"]),
                qty=row["qty"],
                buyer=row["buyer_name"],
                seller=row["seller_name"],
                buyer_id=row.get("buyer_id"),
                seller_id=row.get("seller_id"),
                buyer_fee=float(row["buyer_fee"]),
                seller_fee=float(row["seller_fee"]),
                platform_revenue=float(row["platform_revenue"]),
                timestamp=time.time(),
            ))
        trades.reverse()
        return trades
    except Exception as e:
        print(f"DB load trades error: {e}")
        return []


def verify_user_token(token: str) -> Optional[dict]:
    """Verify a Supabase JWT and return user info."""
    client = get_db()
    if not client or not token:
        return None
    try:
        user_response = client.auth.get_user(token)
        if user_response and user_response.user:
            user = user_response.user
            # Fetch display name from profiles
            profile = client.table("profiles").select("display_name").eq(
                "id", str(user.id)
            ).single().execute()
            return {
                "id": str(user.id),
                "email": user.email,
                "display_name": profile.data.get("display_name", "Trader") if profile.data else "Trader",
            }
    except Exception as e:
        print(f"Auth verify error: {e}")
    return None


# ── Connection Manager ──────────────────────────────────────────

class ConnectionManager:
    def __init__(self):
        self.active: list[dict] = []  # [{ws, user}]

    async def connect(self, ws: WebSocket, user: Optional[dict] = None):
        await ws.accept()
        self.active.append({"ws": ws, "user": user})

    def disconnect(self, ws: WebSocket):
        self.active = [c for c in self.active if c["ws"] != ws]

    async def broadcast(self, message: dict):
        data = json.dumps(message)
        disconnected = []
        for conn in self.active:
            try:
                await conn["ws"].send_text(data)
            except Exception:
                disconnected.append(conn["ws"])
        for ws in disconnected:
            self.disconnect(ws)

    @property
    def user_count(self) -> int:
        return len(self.active)


# ── App Setup ───────────────────────────────────────────────────

manager = ConnectionManager()
books: dict[str, OrderBook] = {}


def init_books():
    """Create orderbooks and load persisted data from Supabase."""
    for card in CARDS:
        book = OrderBook(card_id=card["id"], fee_bps=150)

        # Load persisted orders
        saved_orders = load_orders_from_db(card["id"])
        for order in saved_orders:
            if order.side == Side.BID:
                book._bids.add(order)
            else:
                book._asks.add(order)
            book._orders[order.order_id] = order

        # Load persisted trades
        saved_trades = load_trades_from_db(card["id"])
        book.trades = saved_trades
        book.total_platform_revenue = sum(t.platform_revenue for t in saved_trades)

        books[card["id"]] = book

    loaded_orders = sum(len(b._bids) + len(b._asks) for b in books.values())
    loaded_trades = sum(len(b.trades) for b in books.values())
    print(f"Loaded {loaded_orders} orders and {loaded_trades} trades from database")


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_books()
    yield


app = FastAPI(title="Onyx Vault Orderbook", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── REST Endpoints ──────────────────────────────────────────────

@app.get("/api/cards")
def get_cards():
    return CARDS

@app.get("/api/book/{card_id}")
def get_book(card_id: str):
    book = books.get(card_id)
    if not book:
        return {"error": "Card not found"}
    return book.snapshot()


# ── WebSocket Endpoint ──────────────────────────────────────────

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    # Accept first, then handle auth via messages
    await manager.connect(ws)
    user = None

    try:
        # Send initial state
        initial = {
            "type": "init",
            "cards": CARDS,
            "books": {cid: b.snapshot() for cid, b in books.items()},
            "user": None,
            "online_count": manager.user_count,
        }
        await ws.send_text(json.dumps(initial))

        while True:
            data = json.loads(await ws.receive_text())
            msg_type = data.get("type")

            # ── Authentication ───────────────────────────────
            if msg_type == "auth":
                token = data.get("token", "")
                user = verify_user_token(token)
                if user:
                    # Update this connection's user info
                    for conn in manager.active:
                        if conn["ws"] == ws:
                            conn["user"] = user
                            break
                    await ws.send_text(json.dumps({
                        "type": "auth_result",
                        "success": True,
                        "user": user,
                    }))
                else:
                    await ws.send_text(json.dumps({
                        "type": "auth_result",
                        "success": False,
                        "error": "Invalid or expired token",
                    }))

            # ── Submit Order ─────────────────────────────────
            elif msg_type == "submit_order":
                card_id = data["card_id"]
                book = books.get(card_id)
                if not book:
                    continue

                trader_name = "Anonymous"
                trader_id = None
                if user:
                    trader_name = user["display_name"]
                    trader_id = user["id"]
                else:
                    trader_name = data.get("trader", "Anonymous")

                order_type = data.get("order_type", "limit")
                if order_type == "market":
                    # Extreme price to sweep the book
                    price = 999999.99 if data["side"] == "BID" else 0.01
                else:
                    price = float(data["price"])

                order = Order(
                    side=Side(data["side"]),
                    price=price,
                    qty=int(data["qty"]),
                    trader=trader_name,
                    trader_id=trader_id,
                )
                new_trades = book.submit_order(order)

                # Persist to database
                save_order_to_db(order, card_id)
                for trade in new_trades:
                    save_trade_to_db(trade)
                    # Update the filled resting orders in DB too
                    for o in list(book._orders.values()) + [order]:
                        if o.filled_qty > 0:
                            save_order_to_db(o, card_id)

                # Broadcast to all
                update = {
                    "type": "book_update",
                    "card_id": card_id,
                    "book": book.snapshot(),
                    "new_trades": [t.to_dict() for t in new_trades],
                    "online_count": manager.user_count,
                }
                await manager.broadcast(update)

            # ── Cancel Order ─────────────────────────────────
            elif msg_type == "cancel_order":
                card_id = data["card_id"]
                book = books.get(card_id)
                if not book:
                    continue

                cancelled = book.cancel_order(data["order_id"])
                if cancelled:
                    save_order_to_db(cancelled, card_id)

                update = {
                    "type": "book_update",
                    "card_id": card_id,
                    "book": book.snapshot(),
                    "new_trades": [],
                    "online_count": manager.user_count,
                }
                await manager.broadcast(update)

    except WebSocketDisconnect:
        manager.disconnect(ws)
        # Broadcast updated user count
        try:
            await manager.broadcast({
                "type": "user_count",
                "online_count": manager.user_count,
            })
        except Exception:
            pass


# ── Run ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
