/**
 * ============================================================================
 * AGI Unified Framework - Security & Encryption Module
 * ============================================================================
 * 
 * 存储加密安全模块 - 完整的数据加密、解密、密钥管理和安全存储功能
 * 支持多种加密算法、密钥派生、安全随机数生成
 * 
 * @module persistence-security
 * @version 1.0.0
 * @author AGI Unified Framework Team
 * @license MIT
 * ============================================================================
 */

(function(global) {
    'use strict';

    // =========================================================================
    // Security Error Classes
    // =========================================================================

    class SecurityError extends Error {
        constructor(message, operation, cause = null) {
            super(message);
            this.name = 'SecurityError';
            this.operation = operation;
            this.cause = cause;
            this.timestamp = Date.now();
        }

        toJSON() {
            return {
                name: this.name,
                message: this.message,
                operation: this.operation,
                cause: this.cause?.message || this.cause,
                timestamp: this.timestamp
            };
        }
    }

    class EncryptionError extends SecurityError {
        constructor(message, algorithm, cause = null) {
            super(message, 'encryption', cause);
            this.name = 'EncryptionError';
            this.algorithm = algorithm;
        }
    }

    class DecryptionError extends SecurityError {
        constructor(message, algorithm, cause = null) {
            super(message, 'decryption', cause);
            this.name = 'DecryptionError';
            this.algorithm = algorithm;
        }
    }

    class KeyError extends SecurityError {
        constructor(message, keyId, cause = null) {
            super(message, 'key_operation', cause);
            this.name = 'KeyError';
            this.keyId = keyId;
        }
    }

    // =========================================================================
    // Encryption Algorithms
    // =========================================================================

    const EncryptionAlgorithm = {
        AES_GCM: 'AES-GCM',
        AES_CBC: 'AES-CBC',
        AES_CTR: 'AES-CTR',
        RSA_OAEP: 'RSA-OAEP',
        CHACHA20_POLY1305: 'ChaCha20-Poly1305'
    };

    const KeyDerivationAlgorithm = {
        PBKDF2: 'PBKDF2',
        HKDF: 'HKDF',
        SCRYPT: 'scrypt',
        ARGON2: 'argon2'
    };

    const HashAlgorithm = {
        SHA256: 'SHA-256',
        SHA384: 'SHA-384',
        SHA512: 'SHA-512',
        SHA1: 'SHA-1'
    };

    // =========================================================================
    // Secure Random Generator
    // =========================================================================

    class SecureRandom {
        static generateBytes(length) {
            if (typeof crypto !== 'undefined' && crypto.getRandomValues) {
                return crypto.getRandomValues(new Uint8Array(length));
            }
            
            // Fallback for environments without crypto
            const bytes = new Uint8Array(length);
            for (let i = 0; i < length; i++) {
                bytes[i] = Math.floor(Math.random() * 256);
            }
            return bytes;
        }

        static generateHex(length) {
            const bytes = this.generateBytes(length);
            return Array.from(bytes)
                .map(b => b.toString(16).padStart(2, '0'))
                .join('');
        }

        static generateBase64(length) {
            const bytes = this.generateBytes(length);
            return this.bytesToBase64(bytes);
        }

        static generateUUID() {
            const bytes = this.generateBytes(16);
            bytes[6] = (bytes[6] & 0x0f) | 0x40; // Version 4
            bytes[8] = (bytes[8] & 0x3f) | 0x80; // Variant
            
            const hex = Array.from(bytes)
                .map(b => b.toString(16).padStart(2, '0'))
                .join('');
            
            return `${hex.substr(0, 8)}-${hex.substr(8, 4)}-${hex.substr(12, 4)}-${hex.substr(16, 4)}-${hex.substr(20)}`;
        }

        static generateInt(min, max) {
            const range = max - min;
            const bytesNeeded = Math.ceil(Math.log2(range) / 8);
            const mask = Math.pow(2, bytesNeeded * 8) - 1;
            
            let result;
            do {
                const bytes = this.generateBytes(bytesNeeded);
                result = 0;
                for (let i = 0; i < bytesNeeded; i++) {
                    result = (result << 8) + bytes[i];
                }
                result = result & mask;
            } while (result >= range);
            
            return min + result;
        }

        static bytesToBase64(bytes) {
            const binString = Array.from(bytes, (byte) =>
                String.fromCharCode(byte),
            ).join('');
            return btoa(binString);
        }

        static base64ToBytes(base64) {
            const binString = atob(base64);
            return Uint8Array.from(binString, (m) => m.codePointAt(0));
        }
    }

    // =========================================================================
    // Key Derivation
    // =========================================================================

    class KeyDeriver {
        constructor() {
            this.subtle = typeof crypto !== 'undefined' ? crypto.subtle : null;
        }

        async deriveKeyPBKDF2(password, salt, iterations = 100000, keyLength = 256) {
            if (!this.subtle) {
                throw new SecurityError('Web Crypto API not available', 'derive_key');
            }

            try {
                const encoder = new TextEncoder();
                const passwordBuffer = encoder.encode(password);
                const saltBuffer = typeof salt === 'string' 
                    ? encoder.encode(salt) 
                    : salt;

                const baseKey = await this.subtle.importKey(
                    'raw',
                    passwordBuffer,
                    { name: 'PBKDF2' },
                    false,
                    ['deriveBits', 'deriveKey']
                );

                const derivedKey = await this.subtle.deriveKey(
                    {
                        name: 'PBKDF2',
                        salt: saltBuffer,
                        iterations: iterations,
                        hash: HashAlgorithm.SHA256
                    },
                    baseKey,
                    { name: EncryptionAlgorithm.AES_GCM, length: keyLength },
                    true,
                    ['encrypt', 'decrypt']
                );

                return derivedKey;

            } catch (error) {
                throw new KeyError(
                    `PBKDF2 key derivation failed: ${error.message}`,
                    null,
                    error
                );
            }
        }

        async deriveBitsPBKDF2(password, salt, iterations = 100000, bits = 256) {
            if (!this.subtle) {
                throw new SecurityError('Web Crypto API not available', 'derive_bits');
            }

            try {
                const encoder = new TextEncoder();
                const passwordBuffer = encoder.encode(password);
                const saltBuffer = typeof salt === 'string' 
                    ? encoder.encode(salt) 
                    : salt;

                const baseKey = await this.subtle.importKey(
                    'raw',
                    passwordBuffer,
                    { name: 'PBKDF2' },
                    false,
                    ['deriveBits']
                );

                const derivedBits = await this.subtle.deriveBits(
                    {
                        name: 'PBKDF2',
                        salt: saltBuffer,
                        iterations: iterations,
                        hash: HashAlgorithm.SHA256
                    },
                    baseKey,
                    bits
                );

                return new Uint8Array(derivedBits);

            } catch (error) {
                throw new KeyError(
                    `PBKDF2 bits derivation failed: ${error.message}`,
                    null,
                    error
                );
            }
        }

        async deriveKeyHKDF(ikm, salt, info, keyLength = 256) {
            if (!this.subtle) {
                throw new SecurityError('Web Crypto API not available', 'derive_key');
            }

            try {
                const ikmBuffer = typeof ikm === 'string' 
                    ? new TextEncoder().encode(ikm) 
                    : ikm;
                const saltBuffer = typeof salt === 'string'
                    ? new TextEncoder().encode(salt)
                    : salt;
                const infoBuffer = typeof info === 'string'
                    ? new TextEncoder().encode(info)
                    : info;

                const baseKey = await this.subtle.importKey(
                    'raw',
                    ikmBuffer,
                    { name: 'HKDF' },
                    false,
                    ['deriveKey']
                );

                const derivedKey = await this.subtle.deriveKey(
                    {
                        name: 'HKDF',
                        salt: saltBuffer,
                        info: infoBuffer,
                        hash: HashAlgorithm.SHA256
                    },
                    baseKey,
                    { name: EncryptionAlgorithm.AES_GCM, length: keyLength },
                    true,
                    ['encrypt', 'decrypt']
                );

                return derivedKey;

            } catch (error) {
                throw new KeyError(
                    `HKDF key derivation failed: ${error.message}`,
                    null,
                    error
                );
            }
        }

        // Fallback PBKDF2 implementation for environments without Web Crypto
        deriveKeyPBKDF2Fallback(password, salt, iterations = 100000, keyLength = 32) {
            // Simple fallback - not cryptographically secure
            // In production, use a proper PBKDF2 library
            const encoder = new TextEncoder();
            let key = encoder.encode(password + salt);
            
            for (let i = 0; i < iterations; i++) {
                // Simple mixing
                const mixed = new Uint8Array(key.length);
                for (let j = 0; j < key.length; j++) {
                    mixed[j] = key[j] ^ key[(j + 1) % key.length] ^ (i & 0xFF);
                }
                key = mixed;
            }
            
            return key.slice(0, keyLength);
        }
    }

    // =========================================================================
    // Encryption Service
    // =========================================================================

    class EncryptionService {
        constructor() {
            this.subtle = typeof crypto !== 'undefined' ? crypto.subtle : null;
            this.keyDeriver = new KeyDeriver();
        }

        async generateKey(algorithm = EncryptionAlgorithm.AES_GCM, keyLength = 256, extractable = true) {
            if (!this.subtle) {
                throw new SecurityError('Web Crypto API not available', 'generate_key');
            }

            try {
                const key = await this.subtle.generateKey(
                    {
                        name: algorithm,
                        length: keyLength
                    },
                    extractable,
                    ['encrypt', 'decrypt']
                );

                return key;

            } catch (error) {
                throw new KeyError(
                    `Key generation failed: ${error.message}`,
                    null,
                    error
                );
            }
        }

        async importKey(keyData, algorithm = EncryptionAlgorithm.AES_GCM, format = 'raw') {
            if (!this.subtle) {
                throw new SecurityError('Web Crypto API not available', 'import_key');
            }

            try {
                const keyBuffer = typeof keyData === 'string'
                    ? SecureRandom.base64ToBytes(keyData)
                    : keyData;

                const key = await this.subtle.importKey(
                    format,
                    keyBuffer,
                    { name: algorithm },
                    true,
                    ['encrypt', 'decrypt']
                );

                return key;

            } catch (error) {
                throw new KeyError(
                    `Key import failed: ${error.message}`,
                    null,
                    error
                );
            }
        }

        async exportKey(key, format = 'raw') {
            if (!this.subtle) {
                throw new SecurityError('Web Crypto API not available', 'export_key');
            }

            try {
                const exported = await this.subtle.exportKey(format, key);
                return new Uint8Array(exported);

            } catch (error) {
                throw new KeyError(
                    `Key export failed: ${error.message}`,
                    null,
                    error
                );
            }
        }

        async encrypt(data, key, algorithm = EncryptionAlgorithm.AES_GCM, additionalData = null) {
            if (!this.subtle) {
                throw new SecurityError('Web Crypto API not available', 'encrypt');
            }

            try {
                const encoder = new TextEncoder();
                const dataBuffer = typeof data === 'string'
                    ? encoder.encode(data)
                    : data;

                let iv;
                let algorithmParams;

                switch (algorithm) {
                    case EncryptionAlgorithm.AES_GCM:
                        iv = SecureRandom.generateBytes(12); // 96-bit IV for GCM
                        algorithmParams = { name: algorithm, iv };
                        if (additionalData) {
                            algorithmParams.additionalData = typeof additionalData === 'string'
                                ? encoder.encode(additionalData)
                                : additionalData;
                        }
                        break;
                    
                    case EncryptionAlgorithm.AES_CBC:
                        iv = SecureRandom.generateBytes(16); // 128-bit IV for CBC
                        algorithmParams = { name: algorithm, iv };
                        break;
                    
                    case EncryptionAlgorithm.AES_CTR:
                        iv = SecureRandom.generateBytes(16);
                        algorithmParams = { 
                            name: algorithm, 
                            counter: iv,
                            length: 128 
                        };
                        break;
                    
                    default:
                        throw new EncryptionError(`Unsupported algorithm: ${algorithm}`, algorithm);
                }

                const encrypted = await this.subtle.encrypt(
                    algorithmParams,
                    key,
                    dataBuffer
                );

                // Combine IV + ciphertext
                const result = new Uint8Array(iv.length + encrypted.byteLength);
                result.set(iv);
                result.set(new Uint8Array(encrypted), iv.length);

                return result;

            } catch (error) {
                throw new EncryptionError(
                    `Encryption failed: ${error.message}`,
                    algorithm,
                    error
                );
            }
        }

        async decrypt(encryptedData, key, algorithm = EncryptionAlgorithm.AES_GCM, additionalData = null) {
            if (!this.subtle) {
                throw new SecurityError('Web Crypto API not available', 'decrypt');
            }

            try {
                let ivLength;
                switch (algorithm) {
                    case EncryptionAlgorithm.AES_GCM:
                        ivLength = 12;
                        break;
                    case EncryptionAlgorithm.AES_CBC:
                    case EncryptionAlgorithm.AES_CTR:
                        ivLength = 16;
                        break;
                    default:
                        throw new DecryptionError(`Unsupported algorithm: ${algorithm}`, algorithm);
                }

                // Extract IV and ciphertext
                const iv = encryptedData.slice(0, ivLength);
                const ciphertext = encryptedData.slice(ivLength);

                let algorithmParams;
                switch (algorithm) {
                    case EncryptionAlgorithm.AES_GCM:
                        algorithmParams = { name: algorithm, iv };
                        if (additionalData) {
                            const encoder = new TextEncoder();
                            algorithmParams.additionalData = typeof additionalData === 'string'
                                ? encoder.encode(additionalData)
                                : additionalData;
                        }
                        break;
                    
                    case EncryptionAlgorithm.AES_CBC:
                        algorithmParams = { name: algorithm, iv };
                        break;
                    
                    case EncryptionAlgorithm.AES_CTR:
                        algorithmParams = { 
                            name: algorithm, 
                            counter: iv,
                            length: 128 
                        };
                        break;
                }

                const decrypted = await this.subtle.decrypt(
                    algorithmParams,
                    key,
                    ciphertext
                );

                return new Uint8Array(decrypted);

            } catch (error) {
                throw new DecryptionError(
                    `Decryption failed: ${error.message}`,
                    algorithm,
                    error
                );
            }
        }

        async encryptString(plaintext, key, algorithm = EncryptionAlgorithm.AES_GCM) {
            const encrypted = await this.encrypt(plaintext, key, algorithm);
            return SecureRandom.bytesToBase64(encrypted);
        }

        async decryptString(ciphertext, key, algorithm = EncryptionAlgorithm.AES_GCM) {
            const encrypted = SecureRandom.base64ToBytes(ciphertext);
            const decrypted = await this.decrypt(encrypted, key, algorithm);
            const decoder = new TextDecoder();
            return decoder.decode(decrypted);
        }

        // Password-based encryption
        async encryptWithPassword(data, password, options = {}) {
            const salt = options.salt || SecureRandom.generateBytes(16);
            const iterations = options.iterations || 100000;
            const algorithm = options.algorithm || EncryptionAlgorithm.AES_GCM;

            const key = await this.keyDeriver.deriveKeyPBKDF2(
                password,
                salt,
                iterations,
                256
            );

            const encrypted = await this.encrypt(data, key, algorithm);

            // Combine salt + iterations + encrypted data
            const iterationsBuffer = new Uint8Array(4);
            new DataView(iterationsBuffer.buffer).setUint32(0, iterations, false);

            const result = new Uint8Array(
                salt.length + iterationsBuffer.length + encrypted.length
            );
            result.set(salt);
            result.set(iterationsBuffer, salt.length);
            result.set(encrypted, salt.length + iterationsBuffer.length);

            return result;
        }

        async decryptWithPassword(encryptedData, password, options = {}) {
            const algorithm = options.algorithm || EncryptionAlgorithm.AES_GCM;

            // Extract salt, iterations, and encrypted content
            const salt = encryptedData.slice(0, 16);
            const iterationsBuffer = encryptedData.slice(16, 20);
            const iterations = new DataView(iterationsBuffer.buffer).getUint32(0, false);
            const ciphertext = encryptedData.slice(20);

            const key = await this.keyDeriver.deriveKeyPBKDF2(
                password,
                salt,
                iterations,
                256
            );

            return await this.decrypt(ciphertext, key, algorithm);
        }

        // Simple XOR encryption (for basic obfuscation only - NOT secure)
        xorEncrypt(data, key) {
            const dataBytes = typeof data === 'string' 
                ? new TextEncoder().encode(data) 
                : data;
            const keyBytes = typeof key === 'string'
                ? new TextEncoder().encode(key)
                : key;

            const result = new Uint8Array(dataBytes.length);
            for (let i = 0; i < dataBytes.length; i++) {
                result[i] = dataBytes[i] ^ keyBytes[i % keyBytes.length];
            }

            return result;
        }

        xorDecrypt(encryptedData, key) {
            // XOR is symmetric
            return this.xorEncrypt(encryptedData, key);
        }
    }

    // =========================================================================
    // Hash Service
    // =========================================================================

    class HashService {
        constructor() {
            this.subtle = typeof crypto !== 'undefined' ? crypto.subtle : null;
        }

        async hash(data, algorithm = HashAlgorithm.SHA256) {
            if (!this.subtle) {
                // Fallback to simple hash
                return this.fallbackHash(data);
            }

            try {
                const encoder = new TextEncoder();
                const dataBuffer = typeof data === 'string'
                    ? encoder.encode(data)
                    : data;

                const hashBuffer = await this.subtle.digest(algorithm, dataBuffer);
                return new Uint8Array(hashBuffer);

            } catch (error) {
                throw new SecurityError(
                    `Hashing failed: ${error.message}`,
                    'hash',
                    error
                );
            }
        }

        async hashHex(data, algorithm = HashAlgorithm.SHA256) {
            const hash = await this.hash(data, algorithm);
            return Array.from(hash)
                .map(b => b.toString(16).padStart(2, '0'))
                .join('');
        }

        async hashBase64(data, algorithm = HashAlgorithm.SHA256) {
            const hash = await this.hash(data, algorithm);
            return SecureRandom.bytesToBase64(hash);
        }

        async hmac(data, key, algorithm = HashAlgorithm.SHA256) {
            if (!this.subtle) {
                throw new SecurityError('Web Crypto API not available', 'hmac');
            }

            try {
                const encoder = new TextEncoder();
                const dataBuffer = typeof data === 'string'
                    ? encoder.encode(data)
                    : data;

                const cryptoKey = await this.subtle.importKey(
                    'raw',
                    typeof key === 'string' ? encoder.encode(key) : key,
                    { name: 'HMAC', hash: algorithm },
                    false,
                    ['sign']
                );

                const signature = await this.subtle.sign('HMAC', cryptoKey, dataBuffer);
                return new Uint8Array(signature);

            } catch (error) {
                throw new SecurityError(
                    `HMAC failed: ${error.message}`,
                    'hmac',
                    error
                );
            }
        }

        fallbackHash(data) {
            // Simple hash function for fallback
            const str = typeof data === 'string' ? data : new TextDecoder().decode(data);
            let hash = 0;
            for (let i = 0; i < str.length; i++) {
                const char = str.charCodeAt(i);
                hash = ((hash << 5) - hash) + char;
                hash = hash & hash;
            }
            
            // Convert to 32-byte array
            const result = new Uint8Array(32);
            for (let i = 0; i < 32; i++) {
                result[i] = (hash >> (i % 4) * 8) & 0xFF;
            }
            return result;
        }
    }

    // =========================================================================
    // Secure Storage
    // =========================================================================

    class SecureStorage {
        constructor(storage, encryptionService, options = {}) {
            this.storage = storage;
            this.encryption = encryptionService;
            this.options = {
                keyPrefix: '__secure_',
                algorithm: EncryptionAlgorithm.AES_GCM,
                masterKey: null,
                ...options
            };
            this.keyCache = new Map();
        }

        _getKey(key) {
            return `${this.options.keyPrefix}${key}`;
        }

        async set(key, value, encrypt = true) {
            const storageKey = this._getKey(key);
            
            if (!encrypt || !this.options.masterKey) {
                await this.storage.set(storageKey, value);
                return;
            }

            const serialized = JSON.stringify(value);
            const encrypted = await this.encryption.encryptString(
                serialized,
                this.options.masterKey,
                this.options.algorithm
            );

            await this.storage.set(storageKey, {
                __encrypted: true,
                data: encrypted,
                algorithm: this.options.algorithm
            });
        }

        async get(key, defaultValue = null) {
            const storageKey = this._getKey(key);
            const value = await this.storage.get(storageKey);

            if (value === null || value === undefined) {
                return defaultValue;
            }

            if (value && value.__encrypted && this.options.masterKey) {
                try {
                    const decrypted = await this.encryption.decryptString(
                        value.data,
                        this.options.masterKey,
                        value.algorithm || this.options.algorithm
                    );
                    return JSON.parse(decrypted);
                } catch (error) {
                    console.error('Decryption failed:', error);
                    return defaultValue;
                }
            }

            return value;
        }

        async delete(key) {
            const storageKey = this._getKey(key);
            return await this.storage.delete(storageKey);
        }

        async has(key) {
            const storageKey = this._getKey(key);
            return await this.storage.has(storageKey);
        }

        async keys() {
            const allKeys = await this.storage.keys?.() || [];
            return allKeys
                .filter(k => k.startsWith(this.options.keyPrefix))
                .map(k => k.slice(this.options.keyPrefix.length));
        }

        async clear() {
            const keys = await this.keys();
            for (const key of keys) {
                await this.delete(key);
            }
        }
    }

    // =========================================================================
    // Key Manager
    // =========================================================================

    class KeyManager {
        constructor(storage) {
            this.storage = storage;
            this.keys = new Map();
            this.keyPrefix = '__key_';
        }

        async generateKey(name, options = {}) {
            const encryptionService = new EncryptionService();
            
            const key = await encryptionService.generateKey(
                options.algorithm || EncryptionAlgorithm.AES_GCM,
                options.length || 256
            );

            // Export key for storage
            const exported = await encryptionService.exportKey(key);
            const keyData = {
                name,
                algorithm: options.algorithm || EncryptionAlgorithm.AES_GCM,
                length: options.length || 256,
                key: SecureRandom.bytesToBase64(exported),
                createdAt: Date.now(),
                expiresAt: options.expiresIn 
                    ? Date.now() + options.expiresIn 
                    : null
            };

            // Store encrypted if password provided
            if (options.password) {
                const encrypted = await encryptionService.encryptWithPassword(
                    JSON.stringify(keyData),
                    options.password
                );
                await this.storage.set(`${this.keyPrefix}${name}`, {
                    encrypted: true,
                    data: SecureRandom.bytesToBase64(encrypted)
                });
            } else {
                await this.storage.set(`${this.keyPrefix}${name}`, keyData);
            }

            this.keys.set(name, key);
            return key;
        }

        async getKey(name, password = null) {
            // Check cache
            if (this.keys.has(name)) {
                return this.keys.get(name);
            }

            // Load from storage
            const stored = await this.storage.get(`${this.keyPrefix}${name}`);
            if (!stored) {
                return null;
            }

            let keyData = stored;

            // Decrypt if needed
            if (stored.encrypted && password) {
                const encryptionService = new EncryptionService();
                const encrypted = SecureRandom.base64ToBytes(stored.data);
                const decrypted = await encryptionService.decryptWithPassword(
                    encrypted,
                    password
                );
                keyData = JSON.parse(new TextDecoder().decode(decrypted));
            }

            // Import key
            const encryptionService = new EncryptionService();
            const key = await encryptionService.importKey(
                SecureRandom.base64ToBytes(keyData.key),
                keyData.algorithm
            );

            this.keys.set(name, key);
            return key;
        }

        async deleteKey(name) {
            this.keys.delete(name);
            await this.storage.delete(`${this.keyPrefix}${name}`);
        }

        async rotateKey(name, newPassword = null) {
            // Generate new key
            const oldKey = await this.getKey(name);
            const newKey = await this.generateKey(name, {
                algorithm: EncryptionAlgorithm.AES_GCM,
                length: 256,
                password: newPassword
            });

            return {
                oldKey,
                newKey,
                rotatedAt: Date.now()
            };
        }

        listKeys() {
            return Array.from(this.keys.keys());
        }
    }

    // =========================================================================
    // Export
    // =========================================================================

    const SecuritySystem = {
        // Errors
        SecurityError,
        EncryptionError,
        DecryptionError,
        KeyError,
        
        // Constants
        EncryptionAlgorithm,
        KeyDerivationAlgorithm,
        HashAlgorithm,
        
        // Classes
        SecureRandom,
        KeyDeriver,
        EncryptionService,
        HashService,
        SecureStorage,
        KeyManager
    };

    // Node.js / ES Module support
    if (typeof module !== 'undefined' && module.exports) {
        module.exports = SecuritySystem;
    }

    // AMD support
    if (typeof define === 'function' && define.amd) {
        define('persistence-security', [], function() {
            return SecuritySystem;
        });
    }

    // Global export
    global.PersistenceSecurity = SecuritySystem;

})(typeof window !== 'undefined' ? window : typeof global !== 'undefined' ? global : this);
