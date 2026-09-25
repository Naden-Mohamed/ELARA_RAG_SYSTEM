# test_global_limit.py
import time
import urllib.error
import urllib.request


def test_rate_limit():
    print("Testing global rate limit...")

    print("\nTesting rate limit with test endpoint:")

    for i in range(1, 100):
        try:
            with urllib.request.urlopen("http://127.0.0.1:8000/api/health") as response:
                if response.getcode() == 200:
                    print(f"Request {i}: {response.getcode()} Success")
                else:
                    print(
                        f"Request {i}: {response.getcode()} {response.read().decode()}"
                    )
        except urllib.error.HTTPError as e:
            print(f"Request {i}: {e.code} {e.read().decode()}")
        except Exception as e:
            print(f"Request {i} failed: {e!s}")

        time.sleep(0.01)


if __name__ == "__main__":
    test_rate_limit()
