# OmniAuction Backend

This is the backend service for OmniAuction, containing only the FastAPI application and required dependencies.

## Deployment to Render

1. Push this directory to a new GitHub repository
2. Go to [Render Dashboard](https://dashboard.render.com/)
3. Click "New" and select "Web Service"
4. Connect your GitHub account and select the repository
5. Use the following settings:
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn api.main:app --host 0.0.0.0 --port $PORT`
6. Add environment variables if needed
7. Click "Create Web Service"

## API Documentation

Once deployed, access the API documentation at:
- Interactive docs: `https://your-service-name.onrender.com/docs`
- Alternative docs: `https://your-service-name.onrender.com/redoc`

## WebSocket Endpoint

For real-time updates, connect to:
```
wss://your-service-name.onrender.com/ws
```
