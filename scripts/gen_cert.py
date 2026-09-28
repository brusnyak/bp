"""Generate a self-signed localhost certificate (certs/cert.pem, certs/key.pem) with the
`cryptography` package - no openssl binary needed, same result on Windows/macOS/Linux."""
import datetime
import ipaddress
from pathlib import Path

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID

out = Path(__file__).resolve().parent.parent / "certs"
out.mkdir(exist_ok=True)
key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "localhost")])
now = datetime.datetime.now(datetime.timezone.utc)
cert = (
    x509.CertificateBuilder()
    .subject_name(name).issuer_name(name).public_key(key.public_key())
    .serial_number(x509.random_serial_number())
    .not_valid_before(now - datetime.timedelta(minutes=5))
    .not_valid_after(now + datetime.timedelta(days=365))
    .add_extension(x509.SubjectAlternativeName([x509.DNSName("localhost"), x509.IPAddress(ipaddress.ip_address("127.0.0.1"))]), critical=False)
    .sign(key, hashes.SHA256())
)
(out / "cert.pem").write_bytes(cert.public_bytes(serialization.Encoding.PEM))
key_path = out / "key.pem"
key_path.write_bytes(key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.TraditionalOpenSSL, serialization.NoEncryption()))
try:
    key_path.chmod(0o600)  # no-op on Windows
except OSError:
    pass
print(f"wrote {out / 'cert.pem'} and {key_path}")
