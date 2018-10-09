'''

Implementação com sockets para a comunicação cliente/servidor.

'''

import socket
import os

import datetime

from cryptography import x509
from cryptography.x509.oid import NameOID
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives.serialization import load_pem_public_key
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import dh
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives import serialization

RECV_BUFFER = 4096


class Server:
    def __init__(self, suites):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.suites = suites
        self.cipher = None
        self.key = None
        self.iv = None
        self.encryptor = None
        self.decryptor = None
        self.client_public_key = None
        self.shared_key = None
        self.private_key = None
        self.public_key = None

        self.subject = x509.Name([
            x509.NameAttribute(NameOID.COUNTRY_NAME, u'PT'),
            x509.NameAttribute(NameOID.STATE_OR_PROVINCE_NAME, u'Braga'),
            x509.NameAttribute(NameOID.LOCALITY_NAME, u'Gualtar'),
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, u'Joao Fernandes')
        ])

        self.issuer = self.subject

    def sendSuites(self, clientsocket):
        data = ','.join(self.suites)
        data = data.encode('utf8')
        clientsocket.send(data)

    def TripleDES(self, key=24, iv=8):
        self.key = os.urandom(key)
        self.iv = os.urandom(iv)

        cipher = Cipher(algorithms.TripleDES(self.key), modes.CBC(self.iv), backend=default_backend())

        self.encryptor = cipher.encryptor()
        self.decryptor = cipher.decryptor()

    def AES(self, key=16, iv=16):
        self.key = os.urandom(key)
        self.iv = os.urandom(iv)

        cipher = Cipher(algorithms.AES(self.key), modes.CBC(self.iv), backend=default_backend())

        self.encryptor = cipher.encryptor()
        self.decryptor = cipher.decryptor()

    def AES_DH(self, iv, mode=modes.CBC):
        cipher = Cipher(algorithms.AES(self.key), mode(iv), backend=default_backend())

        self.encryptor = cipher.encryptor()
        self.decryptor = cipher.decryptor()

    def KDF(self, k, salt, hash=hashes.SHA256(), length=16):
        return PBKDF2HMAC(
            algorithm=hash,
            length=length,
            salt=salt,
            iterations=100000,
            backend=default_backend()
        ).derive(k)

    def RSA_Client_Pub_Key(self, pem):

        self.client_public_key = load_pem_public_key(
            data=pem,
            backend=default_backend()
        )

    def RSA_encrypt(self, k):

        return self.client_public_key.encrypt(
            k,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA1()),
                algorithm=hashes.SHA1(),
                label=None
            )
        )

    def RSA_certificate(self):
        self.private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048,
            backend=default_backend()
        )

        self.public_key = self.private_key.public_key()

        self.cert = x509.CertificateBuilder().subject_name(
            self.subject
        ).issuer_name(
            self.issuer
        ).public_key(
            self.public_key
        ).serial_number(
            x509.random_serial_number()
        ).not_valid_before(
            datetime.datetime.utcnow()
        ).not_valid_after(
            datetime.datetime.utcnow() + datetime.timedelta(days=2)
        ).add_extension(
            x509.SubjectAlternativeName([x509.DNSName(u'localhost')]),
            critical=False
        ).sign(self.private_key, hashes.SHA256(), default_backend())

    def certificate(self, private_key, public_key):
        self.cert = x509.CertificateBuilder().subject_name(
            self.subject
        ).issuer_name(
            self.issuer
        ).public_key(
            public_key
        ).serial_number(
            x509.random_serial_number()
        ).not_valid_before(
            datetime.datetime.utcnow()
        ).not_valid_after(
            datetime.datetime.utcnow() + datetime.timedelta(days=2)
        ).add_extension(
            x509.SubjectAlternativeName([x509.DNSName(u'localhost')]),
            critical=False
        ).sign(private_key, hashes.SHA256(), default_backend())

    def GCM_encrypt(self, key, plaintext):
        iv = self.iv

        encryptor = Cipher(
            algorithms.AES(key),
            modes.GCM(iv),
            backend=default_backend()
        ).encryptor()

        ciphertext = encryptor.update(plaintext) + encryptor.finalize()

        return (ciphertext, encryptor.tag)

    def GCM_decrypt(self, key, ciphertext, tag):
        decryptor = Cipher(
            algorithms.AES(key),
            modes.GCM(self.iv, tag),
            backend=default_backend()
        ).decryptor()

        return decryptor.update(ciphertext) + decryptor.finalize()

    def main(self):
        self.sock.bind(('localhost', 8889))
        self.sock.listen(1)

        (clientsocket, address) = self.sock.accept()

        helloMessage = clientsocket.recv(RECV_BUFFER).decode('utf8')
        assert helloMessage == 'Hello'

        self.sendSuites(clientsocket)

        suite = clientsocket.recv(RECV_BUFFER).decode('utf8')
        print(suite)

        if suite == 'TLS_RSA_WITH_3DES_EDE_CBC_SHA':

            self.RSA_certificate()

            clientsocket.send('Client_Pub_Key'.encode('utf8'))
            pem = clientsocket.recv(RECV_BUFFER)
            self.RSA_Client_Pub_Key(pem)

            self.TripleDES()
            key = self.RSA_encrypt(self.key)
            iv = self.RSA_encrypt(self.iv)

            clientsocket.send(key)
            okMessage = clientsocket.recv(RECV_BUFFER).decode('utf8')
            assert okMessage == 'ok'

            clientsocket.send(iv)
            okMessage = clientsocket.recv(RECV_BUFFER).decode('utf8')
            assert okMessage == 'ok'

            print('key: ', self.key.hex())


        elif suite == 'TLS_RSA_WITH_AES_128_CBC_SHA':

            self.RSA_certificate()

            clientsocket.send('Client_Pub_Key'.encode('utf8'))
            pem = clientsocket.recv(RECV_BUFFER)
            self.RSA_Client_Pub_Key(pem)

            self.AES()
            key = self.RSA_encrypt(self.key)
            iv = self.RSA_encrypt(self.iv)

            clientsocket.send(key)
            okMessage = clientsocket.recv(RECV_BUFFER).decode('utf8')
            assert okMessage == 'ok'

            clientsocket.send(iv)
            okMessage = clientsocket.recv(RECV_BUFFER).decode('utf8')
            assert okMessage == 'ok'

            print('key: ', self.key.hex())

        elif suite == 'TLS_DHE_DSS_WITH_AES_128_CBC_SHA':
            diffhelman = dh.generate_parameters(generator=2, key_size=1024, backend=default_backend())
            private_key = diffhelman.generate_private_key()
            g = diffhelman.parameter_numbers().g
            p = diffhelman.parameter_numbers().p
            public_key = private_key.public_key()

            parameters = [g, p, public_key.public_numbers().y]
            parameters = [str(x) for x in parameters]
            parameters = ','.join(parameters)
            parameters = parameters.encode('utf8')

            clientsocket.send(parameters)
            y = int(clientsocket.recv(RECV_BUFFER).decode('utf8'))
            peer_public_numbers = dh.DHPublicNumbers(y, diffhelman.parameter_numbers())
            peer_public_key = peer_public_numbers.public_key(default_backend())
            self.shared_key = private_key.exchange(peer_public_key)

            salt = os.urandom(16)
            clientsocket.send(salt)

            self.key = self.KDF(k=self.shared_key, salt=salt, hash=hashes.SHA1())
            print('key: ', self.key)

            iv = clientsocket.recv(RECV_BUFFER)

            self.AES_DH(iv)

        elif suite == 'TLS_ECDH_ECDSA_WITH_AES_128_GCM_SHA256':
            private_key = ec.generate_private_key(curve=ec.SECP384R1(), backend=default_backend())
            public_key = private_key.public_key()

            serialized_pub_key = public_key.public_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PublicFormat.SubjectPublicKeyInfo
            )

            clientsocket.send(serialized_pub_key)

            pem = clientsocket.recv(RECV_BUFFER)

            peer_public_key = serialization.load_pem_public_key(
                pem,
                backend=default_backend()
            )

            self.shared_key = private_key.exchange(ec.ECDH(), peer_public_key)

            salt = os.urandom(16)
            clientsocket.send(salt)

            self.key = self.KDF(k=self.shared_key, salt=salt, hash=hashes.SHA256())
            print('key: ', self.key)

            self.iv = clientsocket.recv(RECV_BUFFER)

        else:
            diffhelman = dh.generate_parameters(generator=2, key_size=1024, backend=default_backend())
            private_key = diffhelman.generate_private_key()
            g = diffhelman.parameter_numbers().g
            p = diffhelman.parameter_numbers().p
            public_key = private_key.public_key()

            parameters = [g, p, public_key.public_numbers().y]
            parameters = [str(x) for x in parameters]
            parameters = ','.join(parameters)
            parameters = parameters.encode('utf8')

            clientsocket.send(parameters)
            y = int(clientsocket.recv(RECV_BUFFER).decode('utf8'))
            peer_public_numbers = dh.DHPublicNumbers(y, diffhelman.parameter_numbers())
            peer_public_key = peer_public_numbers.public_key(default_backend())
            self.shared_key = private_key.exchange(peer_public_key)

            salt = os.urandom(32)
            clientsocket.send(salt)

            self.key = self.KDF(k=self.shared_key, salt=salt, hash=hashes.SHA384(), length=32)
            print('key: ', self.key)

            self.iv = clientsocket.recv(RECV_BUFFER)

        if not (suite == 'TLS_ECDH_ECDSA_WITH_AES_128_GCM_SHA256' or suite == 'TLS_DHE_DSS_WITH_AES_256_GCM_SHA384'):

            message = self.encryptor.update(b'a secret message'
                                            ) + self.encryptor.finalize()
            clientsocket.send(message)
            print('\n', 'chipertext', message)
        else:
            (message, tag) = self.GCM_encrypt(self.key, b'a secret message')
            clientsocket.send(tag)
            data = clientsocket.recv(RECV_BUFFER).decode('utf8')
            assert data == 'ok'
            clientsocket.send(message)

            print('\n', 'chipertext', message)


        clientsocket.close()
        self.sock.close()


if __name__ == '__main__':
    suites = ['TLS_RSA_WITH_3DES_EDE_CBC_SHA', 'TLS_RSA_WITH_AES_128_CBC_SHA', 'TLS_DHE_DSS_WITH_AES_128_CBC_SHA',
              'TLS_ECDH_ECDSA_WITH_AES_128_GCM_SHA256', 'TLS_DHE_DSS_WITH_AES_256_GCM_SHA384']

    Server(suites).main()
