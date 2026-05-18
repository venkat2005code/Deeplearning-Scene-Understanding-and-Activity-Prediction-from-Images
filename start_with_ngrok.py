from pyngrok import ngrok
import time
from app import app

# Set ngrok authtoken
ngrok.set_auth_token("3CMaCa8mEgyLTgbOjTWYXCM8hrn_4fmqP2cah6BQ2tord6YYh")

# Start ngrok tunnel
print("Starting ngrok tunnel...")
public_url = ngrok.connect(5000, "http")
print(f"\n{'='*60}")
print(f"✓ Public URL: {public_url}")
print(f"{'='*60}\n")

# Keep ngrok active
try:
    app.run(debug=True, use_reloader=False)
except KeyboardInterrupt:
    print("\nShutting down...")
    ngrok.disconnect(public_url)
    ngrok.kill()
