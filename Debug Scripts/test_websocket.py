import asyncio
import json
import websockets

# Use your real sceneId and tokenId
SCENE_ID = "pKZNLeOehA3WJuYC"
TOKEN_ID = "MH95ZnoQkuIIIqoL"

HOST = "127.0.0.1"
PORT = 8765


async def handler(websocket):
    print("Python | Foundry connected via WebSocket")

    # Optionally receive the "hello" from Foundry
    try:
        hello_msg = await asyncio.wait_for(websocket.recv(), timeout=2)
        print("Python | Received from Foundry:", hello_msg)
    except asyncio.TimeoutError:
        print("Python | No hello received (that's okay)")

    print("Python | Ready for input.")
    print("Type coordinates as: x y")
    print("Example: 400 600")
    print("Type 'quit' to exit.")

    try:
        while True:
            user_input = input("Enter new coords (x y): ").strip()

            if user_input.lower() in ("quit", "exit"):
                print("Python | Quitting interactive loop.")
                break

            # Parse input
            try:
                x_str, y_str = user_input.split()
                x = int(x_str)
                y = int(y_str)
            except Exception:
                print("Invalid input. Use: x y  (example: 600 400)")
                continue

            payload = {
                "sceneId": SCENE_ID,
                "tokenId": TOKEN_ID,
                "x": x,
                "y": y,
            }

            print("Python | Sending:", payload)
            await websocket.send(json.dumps(payload))

        print("Python | Done. Closing connection.")
    except websockets.ConnectionClosed:
        print("Python | Connection closed by Foundry.")
    except Exception as e:
        print("Python | Error:", e)


async def main():
    async with websockets.serve(handler, HOST, PORT):
        print(f"Python | WebSocket server listening on ws://{HOST}:{PORT}")
        await asyncio.Future()  # run forever


if __name__ == "__main__":
    asyncio.run(main())
