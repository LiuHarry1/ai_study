from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm, HTTPBearer, HTTPAuthorizationCredentials
from passlib.context import CryptContext
from datetime import datetime, timedelta
import jwt

app = FastAPI()

# Secret key and algorithm
SECRET_KEY = "your_secret_key"
ALGORITHM = "HS256"

# Password hashing setup
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Dummy user database
fake_users_db = {
    "john": {
        "username": "john",
        "hashed_password": pwd_context.hash("secret123"),
    }
}

# Token duration
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Auth dependencies
security = HTTPBearer()

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def authenticate_user(username: str, password: str):
    user = fake_users_db.get(username)
    if not user or not verify_password(password, user["hashed_password"]):
        return False
    return user

def create_access_token(data: dict, expires_delta: timedelta = timedelta(minutes=30)):
    to_encode = data.copy()
    expire = datetime.utcnow() + expires_delta
    to_encode.update({"exp": expire})
    token = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return token

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    token = credentials.credentials
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")



# Endpoint to get token
@app.post("/token")
def login(form_data: OAuth2PasswordRequestForm = Depends()):
    user = authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=400, detail="Incorrect username or password"
        )
    token = create_access_token({"sub": user["username"]})
    return {"access_token": token, "token_type": "bearer"}

# Protected route
@app.get("/secure-data")
def get_secure_data(user=Depends(verify_token)):
    return {"message": "This is protected data", "user": user}


if __name__ == "__main__":
    import uvicorn
    # uvicorn.run(app, host="127.0.0.1", port=8000, reload=True)
    uvicorn.run("fastapi_web:app", host="127.0.0.1", port=8000, reload=True)


