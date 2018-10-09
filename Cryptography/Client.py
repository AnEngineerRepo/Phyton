'''
Implementação com sockets para a comunicação cliente/servidor.

'''

import socket
import random as rand
import os
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.serialization import Encoding
from cryptography.hazmat.primitives.serialization import PublicFormat
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import dh
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives import serialization

RECV_BUFFER = 4096


class Client:
    def __init__(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.private_key = None
        self.public_key = None
        self.key = None
        self.iv = None
        self.encryptor = None
        self.decryptor = None
        self.shared_key = None

    def connect(self, host, port):
        self.sock.connect((host, port))

    def chooseSuite(self, data):
        str_data = data.decode('utf8')
        suites = str_data.split(',')
        rand.shuffle(suites)

        return suites[0]

    def getParameters(self, data):
        str_data = data.decode('utf8')
        suites = str_data.split(',')

        return [int(x) for x in suites]

    def RSA_generate(self, key_size=2048):

        self.private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=key_size,
            backend=default_backend()
        )

        self.public_key = self.private_key.public_key()

    def RSA_decrypt(self, k):
        return self.private_key.decrypt(
            k,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA1()),
                algorithm=hashes.SHA1(),
                label=None
            )
        )

    def TripleDES(self, key, iv):
        self.key = key
        self.iv = iv

        cipher = Cipher(algorithms.TripleDES(self.key), modes.CBC(self.iv), backend=default_backend())

        self.encryptor = cipher.encryptor()
        self.decryptor = cipher.decryptor()

    def AES(self, key, iv):
        self.key = key
        self.iv = iv

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
        self.connect('localhost', 8889)

        self.sock.send('Hello'.encode('utf8'))

        data = self.sock.recv(RECV_BUFFER)
        suite = self.chooseSuite(data)

        self.sock.send(suite.encode('utf8'))
        print(suite)

        if suite == 'TLS_RSA_WITH_3DES_EDE_CBC_SHA':

            clientPubKeyRequest = self.sock.recv(RECV_BUFFER).decode('utf8')
            assert clientPubKeyRequest == 'Client_Pub_Key'

            self.RSA_generate()

            pem = self.public_key.public_bytes(
                encoding=Encoding.PEM,
                format=PublicFormat.SubjectPublicKeyInfo
            )

            self.sock.send(pem)

            cipheredkey = self.sock.recv(RECV_BUFFER)
            self.sock.send('ok'.encode('utf8'))
            cipherediv = self.sock.recv((RECV_BUFFER))
            self.sock.send('ok'.encode('utf8'))

            self.TripleDES(self.RSA_decrypt(cipheredkey), self.RSA_decrypt(cipherediv))

            print('key: ', self.key.hex())

        elif suite == 'TLS_RSA_WITH_AES_128_CBC_SHA':
            clientPubKeyRequest = self.sock.recv(RECV_BUFFER).decode('utf8')
            assert clientPubKeyRequest == 'Client_Pub_Key'

            self.RSA_generate()

            pem = self.public_key.public_bytes(
                encoding=Encoding.PEM,
                format=PublicFormat.SubjectPublicKeyInfo
            )

            self.sock.send(pem)

            cipheredkey = self.sock.recv(RECV_BUFFER)
            self.sock.send('ok'.encode('utf8'))
            cipherediv = self.sock.recv((RECV_BUFFER))
            self.sock.send('ok'.encode('utf8'))

            self.AES(self.RSA_decrypt(cipheredkey), self.RSA_decrypt(cipherediv))

            print('key: ', self.key.hex())

        elif suite == 'TLS_DHE_DSS_WITH_AES_128_CBC_SHA':
            parameters = self.sock.recv(RECV_BUFFER)

            parameters = self.getParameters(parameters)

            pn = dh.DHParameterNumbers(parameters[1], parameters[0])
            diffhelman = pn.parameters(default_backend())
            peer_public_numbers = dh.DHPublicNumbers(parameters[2], pn)
            peer_public_key = peer_public_numbers.public_key(default_backend())
            private_key = diffhelman.generate_private_key()
            public_key = private_key.public_key()
            self.shared_key = private_key.exchange(peer_public_key)

            self.sock.send(str(public_key.public_numbers().y).encode('utf8'))

            salt = self.sock.recv(RECV_BUFFER)

            self.key = self.KDF(k=self.shared_key, salt=salt, hash=hashes.SHA1())
            print('key: ', self.key)

            self.iv = os.urandom(16)
            self.sock.send(self.iv)
            self.AES_DH(self.iv)


        elif suite == 'TLS_ECDH_ECDSA_WITH_AES_128_GCM_SHA256':

            private_key = ec.generate_private_key(curve=ec.SECP384R1(), backend=default_backend())
            public_key = private_key.public_key()

            serialized_pub_key = public_key.public_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PublicFormat.SubjectPublicKeyInfo
            )

            pem = self.sock.recv(RECV_BUFFER)
            self.sock.send(serialized_pub_key)

            peer_public_key = serialization.load_pem_public_key(
                pem,
                backend=default_backend()
            )

            self.shared_key = private_key.exchange(ec.ECDH(), peer_public_key)

            salt = self.sock.recv(RECV_BUFFER)

            self.key = self.KDF(k=self.shared_key, salt=salt, hash=hashes.SHA256())
            print('key: ', self.key)

            self.iv = os.urandom(16)
            self.sock.send(self.iv)

        else:
            parameters = self.sock.recv(RECV_BUFFER)

            parameters = self.getParameters(parameters)

            pn = dh.DHParameterNumbers(parameters[1], parameters[0])
            diffhelman = pn.parameters(default_backend())
            peer_public_numbers = dh.DHPublicNumbers(parameters[2], pn)
            peer_public_key = peer_public_numbers.public_key(default_backend())
            private_key = diffhelman.generate_private_key()
            public_key = private_key.public_key()
            self.shared_key = private_key.exchange(peer_public_key)

            self.sock.send(str(public_key.public_numbers().y).encode('utf8'))

            salt = self.sock.recv(RECV_BUFFER)

            self.key = self.KDF(k=self.shared_key, salt=salt, hash=hashes.SHA384(), length=32)
            print('key: ', self.key)

            self.iv = os.urandom(32)
            self.sock.send(self.iv)

        if not (suite == 'TLS_ECDH_ECDSA_WITH_AES_128_GCM_SHA256' or suite == 'TLS_DHE_DSS_WITH_AES_256_GCM_SHA384'):
            cipher = self.sock.recv(RECV_BUFFER)
            message = self.decryptor.update(cipher) + self.decryptor.finalize()
            print('\n', 'plaintext: ', message.decode('utf8'))

        else:
            tag = self.sock.recv(RECV_BUFFER)
            print(tag)
            self.sock.send('ok'.encode('utf8'))
            ciphertext = self.sock.recv(RECV_BUFFER)

            message = self.GCM_decrypt(self.key, ciphertext, tag)
            print('\n', 'plaintext: ', message.decode('utf8'))

        self.sock.close()


if __name__ == '__main__':
    Client().main()
