import http.server
import socketserver
import sqlite3
import json
import os
from datetime import datetime

PORT = 8000
DB_FILE = 'waitlist.db'

# Initialize SQLite waitlist
def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS waitlist
        (id INTEGER PRIMARY KEY AUTOINCREMENT,
         email TEXT UNIQUE NOT NULL,
         timestamp TEXT NOT NULL)
    ''')
    conn.commit()
    conn.close()

class WaitlistHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        # Allow CORS for local dev testing
        self.send_header('Access-Control-Allow-Origin', '*')
        super().end_headers()
        
    def do_POST(self):
        if self.path == '/api/waitlist':
            content_length = int(self.headers.get('Content-Length', 0))
            if content_length > 0:
                body = self.rfile.read(content_length)
                try:
                    data = json.loads(body)
                    email = data.get('email')
                    
                    if not email or '@' not in email:
                        self.send_response(400)
                        self.send_header('Content-Type', 'application/json')
                        self.end_headers()
                        self.wfile.write(json.dumps({'error': 'Invalid email address'}).encode())
                        return
                    
                    # Store to DB
                    conn = sqlite3.connect(DB_FILE)
                    c = conn.cursor()
                    try:
                        c.execute("INSERT INTO waitlist (email, timestamp) VALUES (?, ?)", 
                                  (email, datetime.utcnow().isoformat()))
                        conn.commit()
                        self.send_response(200)
                        self.send_header('Content-Type', 'application/json')
                        self.end_headers()
                        self.wfile.write(json.dumps({'message': 'Success'}).encode())
                    except sqlite3.IntegrityError:
                        self.send_response(400)
                        self.send_header('Content-Type', 'application/json')
                        self.end_headers()
                        self.wfile.write(json.dumps({'error': 'Email already registered'}).encode())
                    finally:
                        conn.close()
                except Exception as e:
                    self.send_response(500)
                    self.send_header('Content-Type', 'application/json')
                    self.end_headers()
                    self.wfile.write(json.dumps({'error': str(e)}).encode())
            else:
                self.send_response(400)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({'error': 'Empty request'}).encode())
        else:
            self.send_response(404)
            self.end_headers()

if __name__ == "__main__":
    init_db()
    with socketserver.TCPServer(("", PORT), WaitlistHandler) as httpd:
        print(f"Serving at http://localhost:{PORT}")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down server.")
