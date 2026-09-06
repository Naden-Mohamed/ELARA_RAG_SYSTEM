from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from core.security import (
    create_access_token,
    decode_access_token,
    get_password_hash,
    verify_password,
)
from db.user_model import UserModel
from models.api_responce import APIResponce
from routers.schemas.auth_schemas import (
    TokenResponse,
    UserLoginRequest,
    UserRegisterRequest,
)

auth_router = APIRouter(tags=["Authentication"], prefix="/auth")
security = HTTPBearer()


# Dependency to get the current authenticated user
async def get_current_user(
    request: Request, creds: HTTPAuthorizationCredentials = Depends(security)
):
    payload = decode_access_token(creds.credentials)
    if not payload or "sub" not in payload:
        raise HTTPException(
            status_code=401, detail="Invalid or expired authentication credentials."
        )
    user = await UserModel(request.app.state.db_client).get_by_id(payload["sub"])
    if not user:
        raise HTTPException(status_code=404, detail="User not found.")
    return user


@auth_router.post("/register", response_model=APIResponce)
async def register(request: Request, payload: UserRegisterRequest):
    user_model = UserModel(request.app.state.db_client)
    await user_model.init_indexes()

    # Check if user already exists
    existing_user = await user_model.get_by_email(payload.email)
    if existing_user:
        return APIResponce(
            status_code=status.HTTP_400_BAD_REQUEST,
            status="failed",
            error="Email address already registered.",
        )

    # Hash Password and Prepare Document
    user_dict = payload.model_dump()
    raw_password = user_dict.pop("password")
    user_dict["hashed_password"] = get_password_hash(raw_password)

    new_user = await user_model.create_user(user_dict)
    user_id_str = str(new_user["_id"])

    # Generate JWT Token directly on register
    token = create_access_token(
        data={"sub": user_id_str, "persona": new_user["persona"]}
    )

    return APIResponce(
        status_code=status.HTTP_201_CREATED,
        status="success",
        data=TokenResponse(
            access_token=token,
            user_id=user_id_str,
            full_name=new_user["full_name"],
            persona=new_user["persona"],
        ).dict(),
    )


@auth_router.post("/login", response_model=APIResponce)
async def login(request: Request, payload: UserLoginRequest):
    user_model = UserModel(request.app.state.db_client)
    user = await user_model.get_by_email(payload.email)

    if not user or not verify_password(payload.password, user["hashed_password"]):
        return APIResponce(
            status_code=status.HTTP_401_UNAUTHORIZED,
            status="failed",
            error="Invalid email or password.",
        )

    user_id_str = str(user["_id"])
    token = create_access_token(data={"sub": user_id_str, "persona": user["persona"]})

    return APIResponce(
        status_code=status.HTTP_200_OK,
        status="success",
        data=TokenResponse(
            access_token=token,
            user_id=user_id_str,
            full_name=user["full_name"],
            persona=user["persona"],
        ).dict(),
    )


@auth_router.get("/me", response_model=APIResponce)
async def get_my_profile(current_user: dict = Depends(get_current_user)):
    """Returns the authenticated user data including clinical and mother profile."""
    current_user["_id"] = str(current_user["_id"])
    current_user.pop("hashed_password", None)

    return APIResponce(
        status_code=status.HTTP_200_OK, status="success", data=current_user
    )
