from datetime import UTC, date, datetime

from bson import ObjectId
from motor.motor_asyncio import AsyncIOMotorDatabase


def convert_dates_to_datetimes(data: dict) -> dict:
    """Recursively converts any datetime.date instances to datetime.datetime for BSON encoding."""
    for key, value in data.items():
        if isinstance(value, dict):
            convert_dates_to_datetimes(value)
        elif isinstance(value, date) and not isinstance(value, datetime):
            data[key] = datetime.combine(value, datetime.min.time(), tzinfo=UTC)
    return data


class UserModel:
    def __init__(self, db: AsyncIOMotorDatabase):
        self.collection = db["users"]

    async def init_indexes(self):
        await self.collection.create_index("email", unique=True)

    async def get_by_email(self, email: str) -> dict | None:
        return await self.collection.find_one({"email": email.lower().strip()})

    async def get_by_id(self, user_id: str) -> dict | None:
        return await self.collection.find_one({"_id": ObjectId(user_id)})

    async def create_user(self, user_data: dict) -> dict:
        user_data["email"] = user_data["email"].lower().strip()
        user_data["created_at"] = datetime.now(UTC)
        user_data["is_active"] = True

        # Convert nested dates (e.g. expected_due_date) to BSON-compatible datetime
        user_data = convert_dates_to_datetimes(user_data)

        result = await self.collection.insert_one(user_data)
        user_data["_id"] = result.inserted_id
        return user_data

    async def update_mother_profile(self, user_id: str, profile_data: dict) -> bool:
        profile_data = convert_dates_to_datetimes(profile_data)
        res = await self.collection.update_one(
            {"_id": ObjectId(user_id)}, {"$set": {"mother_profile": profile_data}}
        )
        return res.modified_count > 0
