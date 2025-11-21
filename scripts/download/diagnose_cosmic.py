#!/usr/bin/env python3
"""
Test COSMIC API with CORRECT endpoint (v1/downloads/scripted)
Based on official documentation from cancer.sanger.ac.uk
"""

import os
import base64
import requests

# Load from environment
email = os.getenv('COSMIC_EMAIL')
password = os.getenv('COSMIC_PASSWORD')

print(f"Email: {email}")
print(f"Password: {'*' * len(password) if password else 'NOT SET'}")

# CORRECT API endpoint (from official docs)
API_BASE = "https://cancer.sanger.ac.uk/api/mono/products/v1/downloads/scripted"

# File path (lowercase grch38 as per docs)
file_path = "grch38/cosmic/v102/Cosmic_MutantCensus_Tsv_v102_GRCh38.tar"

# Build full URL with query params
url = f"{API_BASE}?path={file_path}&bucket=downloads"

print(f"\nAPI Endpoint: {API_BASE}")
print(f"File path: {file_path}")
print(f"Full URL: {url}")

# Create Base64 auth string (IMPORTANT: include newline as per echo command)
# The docs show: echo 'email:password' | base64
# echo adds a newline, so we need to match that
auth_string = f"{email}:{password}\n"
auth_base64 = base64.b64encode(auth_string.encode()).decode()
print(f"\nAuth (with newline): Basic {auth_base64[:30]}...")

# Also try without newline
auth_string_no_nl = f"{email}:{password}"
auth_base64_no_nl = base64.b64encode(auth_string_no_nl.encode()).decode()
print(f"Auth (no newline):   Basic {auth_base64_no_nl[:30]}...")

print("\n" + "=" * 60)
print("TEST 1: With newline in auth (matches echo behavior)")
print("=" * 60)
headers1 = {"Authorization": f"Basic {auth_base64}"}
response1 = requests.get(url, headers=headers1)
print(f"Status: {response1.status_code}")
print(f"Content-Type: {response1.headers.get('Content-Type', 'N/A')}")
print(f"Response: {response1.text[:500]}")

print("\n" + "=" * 60)
print("TEST 2: Without newline in auth")
print("=" * 60)
headers2 = {"Authorization": f"Basic {auth_base64_no_nl}"}
response2 = requests.get(url, headers=headers2)
print(f"Status: {response2.status_code}")
print(f"Content-Type: {response2.headers.get('Content-Type', 'N/A')}")
print(f"Response: {response2.text[:500]}")

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)

success = False
for i, resp in enumerate([response1, response2], 1):
    if resp.status_code == 200 and 'url' in resp.text.lower():
        print(f"✓ TEST {i} SUCCESS - Got download URL!")
        success = True
        try:
            import json
            data = json.loads(resp.text)
            print(f"  Download URL: {data.get('url', 'N/A')[:80]}...")
        except:
            pass

if not success:
    print("✗ Both tests failed")
    if response1.status_code == 401 or response2.status_code == 401:
        print("  → 401 Unauthorized: Check credentials")
    elif 'login' in response1.text.lower():
        print("  → Got login page: Authentication not working")
