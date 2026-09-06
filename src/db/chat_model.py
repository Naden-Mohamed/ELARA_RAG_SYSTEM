from datetime import UTC, datetime

from bson import ObjectId
from motor.motor_asyncio import AsyncIOMotorDatabase


class ChatModel:
    def __init__(self, db: AsyncIOMotorDatabase):
        self.chats = db["chats"]
        self.messages = db["messages"]
        self.memory = db["clinical_memory"]

    async def init_indexes(self):
        await self.messages.create_index([("chat_id", 1), ("created_at", 1)])
        await self.memory.create_index([("user_id", 1), ("is_active", 1)])

    async def get_or_create_chat(
        self, user_id: str, title: str = "New Consultation"
    ) -> str:
        chat = await self.chats.find_one(
            {"user_id": ObjectId(user_id), "is_archived": False},
            sort=[("updated_at", -1)],
        )
        if chat:
            return str(chat["_id"])

        new_chat = {
            "user_id": ObjectId(user_id),
            "title": title,
            "created_at": datetime.now(UTC),
            "updated_at": datetime.now(UTC),
            "is_archived": False,
        }
        res = await self.chats.insert_one(new_chat)
        return str(res.inserted_id)

    async def get_recent_messages(self, chat_id: str, limit: int = 6) -> list[dict]:
        cursor = (
            self.messages.find({"chat_id": ObjectId(chat_id)})
            .sort("created_at", -1)
            .limit(limit)
        )
        msgs = await cursor.to_list(length=limit)
        msgs.reverse()  # Chronological order
        return [{"role": m["role"], "content": m["content"]} for m in msgs]

    async def add_message(
        self,
        chat_id: str,
        user_id: str,
        role: str,
        content: str,
        citations: list = None,
        latency: float = 0.0,
    ):
        doc = {
            "chat_id": ObjectId(chat_id),
            "user_id": ObjectId(user_id),
            "role": role,
            "content": content,
            "citations": citations or [],
            "latency_seconds": latency,
            "created_at": datetime.now(UTC),
        }
        await self.messages.insert_one(doc)
        await self.chats.update_one(
            {"_id": ObjectId(chat_id)}, {"$set": {"updated_at": datetime.now(UTC)}}
        )

    async def get_active_clinical_memory(self, user_id: str) -> list[str]:
        cursor = self.memory.find({"user_id": ObjectId(user_id), "is_active": True})
        docs = await cursor.to_list(length=20)
        return [f"- {d['category'].upper()}: {d['key']} = {d['value']}" for d in docs]

    async def get_user_chats(self, user_id: str) -> list[dict]:
        """Returns the list of all chat sessions for the sidebar/history drawer."""
        cursor = self.chats.find(
            {"user_id": ObjectId(user_id), "is_archived": False}
        ).sort("updated_at", -1)

        chats = await cursor.to_list(length=50)
        return [
            {
                "chat_id": str(c["_id"]),
                "title": c.get("title", "محادثة استشارية"),
                "updated_at": c["updated_at"],
                "is_archived": c.get("is_archived", False),
            }
            for c in chats
        ]

    async def get_chat_history_paginated(
        self, chat_id: str, user_id: str, page: int = 1, page_size: int = 20
    ) -> dict | None:
        """Returns paginated messages ordered chronologically for rendering."""
        # 1. Verify ownership
        chat = await self.chats.find_one(
            {"_id": ObjectId(chat_id), "user_id": ObjectId(user_id)}
        )
        if not chat:
            return None

        # 2. Count total messages in this chat
        total_messages = await self.messages.count_documents(
            {"chat_id": ObjectId(chat_id)}
        )
        skip_count = (page - 1) * page_size

        # 3. Query messages sorted descending to get the newest page, then reverse for UI display
        cursor = (
            self.messages.find({"chat_id": ObjectId(chat_id)})
            .sort("created_at", -1)
            .skip(skip_count)
            .limit(page_size)
        )

        raw_messages = await cursor.to_list(length=page_size)
        raw_messages.reverse()  # Oldest first within the retrieved window

        messages = [
            {
                "message_id": str(m["_id"]),
                "role": m["role"],
                "content": m["content"],
                "citations": m.get("citations", []),
                "created_at": m["created_at"],
            }
            for m in raw_messages
        ]

        has_more = (skip_count + len(messages)) < total_messages

        return {
            "chat_id": str(chat["_id"]),
            "title": chat.get("title", "محادثة استشارية"),
            "total_messages": total_messages,
            "page": page,
            "page_size": page_size,
            "has_more": has_more,
            "messages": messages,
        }
