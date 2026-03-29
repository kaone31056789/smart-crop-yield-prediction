# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| latest  | :white_check_mark: |

## Reporting a Vulnerability

If you discover a security vulnerability in this project, please report it responsibly.

**Please do NOT open a public GitHub issue for security vulnerabilities.**

Instead, please report security issues by opening a private security advisory via GitHub:

1. Go to the **Security** tab of this repository
2. Click **Report a vulnerability**
3. Provide a detailed description of the vulnerability

### What to include in your report

- Description of the vulnerability
- Steps to reproduce the issue
- Potential impact
- Suggested fix (if any)

### What to expect

- You will receive an acknowledgment within **48 hours**
- We will investigate and provide updates on the progress
- Once resolved, we will credit you in the fix (unless you prefer to remain anonymous)

## Security Best Practices for Contributors

- Never commit API keys, passwords, or other secrets to the repository
- Keep dependencies up to date
- Follow the principle of least privilege when adding new functionality
- Validate and sanitize all user inputs

## Dependencies

This project uses the following external services (all free, no API keys required):

- [Open-Meteo](https://open-meteo.com/) — Weather data
- [Wikipedia REST API](https://en.wikipedia.org/api/rest_v1/) — Crop information
- [IP-API](http://ip-api.com/) — Geolocation

Review `requirements.txt` for the full list of Python dependencies.
