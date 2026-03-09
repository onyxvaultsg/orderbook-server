"""
Onyx Vault — Orderbook WebSocket Server
Analyze. Acquire. Profit.

A FastAPI server that manages shared orderbook state and broadcasts
updates to all connected clients via WebSocket.

Run locally:
    pip install fastapi uvicorn sortedcontainers
    uvicorn server:app --reload --port 8000
"""

from __future__ import annotations

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
from sortedcontainers import SortedList


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
            else:
                buyer, seller = best.trader, aggressor.trader

            trade = Trade(
                trade_id=uuid.uuid4().hex[:8],
                card_id=self.card_id,
                price=fill_price,
                qty=fill_qty,
                buyer=buyer,
                seller=seller,
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
# Edit this list to add/remove/reprice cards.

CARDS = [
    {"id": "SV01-025-PSA10", "name": "Charizard ex SV",    "base_price": 285},
    {"id": "SWSH12-GG70-PSA10", "name": "Giratina V Alt Art", "base_price": 165},
    {"id": "SM12-SV49-PSA10", "name": "Charizard VMAX",    "base_price": 420},
]

TRADERS = ["KantoKid", "JohtoTrader", "HoennVault", "SinnohDeals",
           "UnovaFlips", "KalosKing", "AlolaArb", "GalarGains"]


# ── Connection Manager ──────────────────────────────────────────

class ConnectionManager:
    def __init__(self):
        self.active: list[WebSocket] = []

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.active.append(ws)

    def disconnect(self, ws: WebSocket):
        self.active.remove(ws)

    async def broadcast(self, message: dict):
        data = json.dumps(message)
        disconnected = []
        for ws in self.active:
            try:
                await ws.send_text(data)
            except Exception:
                disconnected.append(ws)
        for ws in disconnected:
            self.active.remove(ws)


# ── App Setup ───────────────────────────────────────────────────

manager = ConnectionManager()
books: dict[str, OrderBook] = {}


def seed_books():
    """Create orderbooks and seed with initial liquidity."""
    import random
    for card in CARDS:
        book = OrderBook(card_id=card["id"], fee_bps=150)
        bp = card["base_price"]
        for i in range(6):
            bid_price = round(bp - 2 - i * (1 + random.random() * 2), 2)
            ask_price = round(bp + 2 + i * (1 + random.random() * 2), 2)
            book.submit_order(Order(
                side=Side.BID, price=bid_price,
                qty=random.randint(1, 4),
                trader=random.choice(TRADERS),
            ))
            book.submit_order(Order(
                side=Side.ASK, price=ask_price,
                qty=random.randint(1, 4),
                trader=random.choice(TRADERS),
            ))
        books[card["id"]] = book


@asynccontextmanager
async def lifespan(app: FastAPI):
    seed_books()
    yield


app = FastAPI(title="Onyx Vault Orderbook", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict to your domain
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
    await manager.connect(ws)

    # Send initial state for all books
    try:
        initial = {
            "type": "init",
            "cards": CARDS,
            "books": {cid: b.snapshot() for cid, b in books.items()},
        }
        await ws.send_text(json.dumps(initial))

        while True:
            data = json.loads(await ws.receive_text())
            msg_type = data.get("type")

            if msg_type == "submit_order":
                card_id = data["card_id"]
                book = books.get(card_id)
                if not book:
                    continue

                order = Order(
                    side=Side(data["side"]),
                    price=float(data["price"]),
                    qty=int(data["qty"]),
                    trader=data.get("trader", "Anonymous"),
                )
                new_trades = book.submit_order(order)

                # Broadcast updated book + any new trades to ALL clients
                update = {
                    "type": "book_update",
                    "card_id": card_id,
                    "book": book.snapshot(),
                    "new_trades": [t.to_dict() for t in new_trades],
                }
                await manager.broadcast(update)

            elif msg_type == "cancel_order":
                card_id = data["card_id"]
                book = books.get(card_id)
                if not book:
                    continue
                book.cancel_order(data["order_id"])
                update = {
                    "type": "book_update",
                    "card_id": card_id,
                    "book": book.snapshot(),
                    "new_trades": [],
                }
                await manager.broadcast(update)

    except WebSocketDisconnect:
        manager.disconnect(ws)


# ── Run ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
