import os, time, base64, httpx
from dotenv import load_dotenv; load_dotenv(".env")
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding
kid=os.environ["KALSHI_API_KEY_ID"]; env=os.environ.get("KALSHI_ENV","demo")
base="https://demo-api.kalshi.co/trade-api/v2" if env=="demo" else "https://api.elections.kalshi.com/trade-api/v2"
key=serialization.load_pem_private_key(open(os.environ["KALSHI_PRIVATE_KEY_PATH"],"rb").read(),password=None)
def headers(method,path):
    ts=str(int(time.time()*1000)); msg=(ts+method+path).encode()
    sig=key.sign(msg,padding.PSS(mgf=padding.MGF1(hashes.SHA256()),salt_length=padding.PSS.DIGEST_LENGTH),hashes.SHA256())
    return {"KALSHI-ACCESS-KEY":kid,"KALSHI-ACCESS-TIMESTAMP":ts,"KALSHI-ACCESS-SIGNATURE":base64.b64encode(sig).decode()}
for path in ("/portfolio/balance","/portfolio/positions"):
    r=httpx.get(base+path,headers=headers("GET","/trade-api/v2"+path),timeout=20)
    print(path,r.status_code,r.text[:160])
