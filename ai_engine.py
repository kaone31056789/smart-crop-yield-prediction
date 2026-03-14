"""
=============================================================================
 SMART CROP YIELD PREDICTION USING ARTIFICIAL INTELLIGENCE
 Module: ai_engine.py — Multi-Provider AI Engine (Free-First Fallback Chain)
=============================================================================

 PURPOSE:
   Central AI gateway that routes requests through multiple LLM/Vision
   providers with automatic fallback. Free APIs tried first, then paid.

   Fallback chain (configurable):
     1. Google Gemini   — Free tier (15 RPM, vision + text)
     2. Groq            — Free tier (fast inference, text only)
     3. Claude API      — Paid (if key provided)
     4. OpenAI API      — Paid (if key provided)
     5. Grok/xAI API    — Paid (if key provided)

 AUTHOR : AgriTech AI Solutions
 VERSION: 4.2.0
=============================================================================
"""

import os
import json
import base64
import requests
import time
from io import BytesIO
from PIL import Image

# ═══════════════════════════════════════════════════════════════════════════
#  PROVIDER CONFIGURATIONS
# ═══════════════════════════════════════════════════════════════════════════

PROVIDERS = {
    "gemini": {
        "name": "Google Gemini (Free)",
        "base_url": "https://generativelanguage.googleapis.com/v1beta",
        "env_key": "GEMINI_API_KEY",
        "supports_vision": True,
        "free_tier": True,
        "model_text": "gemini-2.0-flash",
        "model_vision": "gemini-2.0-flash",
    },
    "groq": {
        "name": "Groq (Free)",
        "base_url": "https://api.groq.com/openai/v1",
        "env_key": "GROQ_API_KEY",
        "supports_vision": True,
        "free_tier": True,
        "model_text": "llama-3.3-70b-versatile",
        "model_vision": "llama-3.2-90b-vision-preview",
    },
    "claude": {
        "name": "Claude (Anthropic)",
        "base_url": "https://api.anthropic.com/v1",
        "env_key": "ANTHROPIC_API_KEY",
        "supports_vision": True,
        "free_tier": False,
        "model_text": "claude-sonnet-4-20250514",
        "model_vision": "claude-sonnet-4-20250514",
    },
    "openai": {
        "name": "ChatGPT (OpenAI)",
        "base_url": "https://api.openai.com/v1",
        "env_key": "OPENAI_API_KEY",
        "supports_vision": True,
        "free_tier": False,
        "model_text": "gpt-4o-mini",
        "model_vision": "gpt-4o-mini",
    },
    "xai": {
        "name": "Grok (xAI)",
        "base_url": "https://api.x.ai/v1",
        "env_key": "XAI_API_KEY",
        "supports_vision": True,
        "free_tier": False,
        "model_text": "grok-3-mini-fast",
        "model_vision": "grok-2-vision",
    },
}

# Default fallback order: free first, then paid
DEFAULT_CHAIN = ["gemini", "groq", "claude", "openai", "xai"]


# ═══════════════════════════════════════════════════════════════════════════
#  IMAGE ENCODING HELPER
# ═══════════════════════════════════════════════════════════════════════════

def _encode_image_base64(image: Image.Image, max_dim: int = 1024) -> str:
    """Resize and encode PIL Image to base64 JPEG."""
    w, h = image.size
    if max(w, h) > max_dim:
        ratio = max_dim / max(w, h)
        image = image.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)
    image = image.convert("RGB")
    buf = BytesIO()
    image.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# ═══════════════════════════════════════════════════════════════════════════
#  PROVIDER-SPECIFIC API CALLS
# ═══════════════════════════════════════════════════════════════════════════

def _call_gemini(api_key: str, prompt: str, image_b64: str = None,
                 model: str = None, system: str = None) -> str:
    """Call Google Gemini API."""
    cfg = PROVIDERS["gemini"]
    model = model or (cfg["model_vision"] if image_b64 else cfg["model_text"])
    url = f"{cfg['base_url']}/models/{model}:generateContent?key={api_key}"

    parts = []
    if image_b64:
        parts.append({
            "inline_data": {
                "mime_type": "image/jpeg",
                "data": image_b64
            }
        })
    parts.append({"text": prompt})

    body = {"contents": [{"parts": parts}]}
    if system:
        body["system_instruction"] = {"parts": [{"text": system}]}
    body["generationConfig"] = {"temperature": 0.3, "maxOutputTokens": 4096}

    resp = requests.post(url, json=body, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    return data["candidates"][0]["content"]["parts"][0]["text"]


def _call_openai_compatible(provider: str, api_key: str, prompt: str,
                            image_b64: str = None, model: str = None,
                            system: str = None) -> str:
    """Call OpenAI-compatible API (OpenAI, Groq, xAI)."""
    cfg = PROVIDERS[provider]
    model = model or (cfg["model_vision"] if image_b64 else cfg["model_text"])
    url = f"{cfg['base_url']}/chat/completions"

    messages = []
    if system:
        messages.append({"role": "system", "content": system})

    if image_b64:
        content = [
            {"type": "image_url", "image_url": {
                "url": f"data:image/jpeg;base64,{image_b64}"
            }},
            {"type": "text", "text": prompt}
        ]
        messages.append({"role": "user", "content": content})
    else:
        messages.append({"role": "user", "content": prompt})

    body = {
        "model": model,
        "messages": messages,
        "temperature": 0.3,
        "max_tokens": 4096,
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    resp = requests.post(url, json=body, headers=headers, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"]


def _call_claude(api_key: str, prompt: str, image_b64: str = None,
                 model: str = None, system: str = None) -> str:
    """Call Anthropic Claude API."""
    cfg = PROVIDERS["claude"]
    model = model or (cfg["model_vision"] if image_b64 else cfg["model_text"])
    url = f"{cfg['base_url']}/messages"

    content = []
    if image_b64:
        content.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/jpeg",
                "data": image_b64
            }
        })
    content.append({"type": "text", "text": prompt})

    body = {
        "model": model,
        "max_tokens": 4096,
        "messages": [{"role": "user", "content": content}]
    }
    if system:
        body["system"] = system

    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "Content-Type": "application/json"
    }

    resp = requests.post(url, json=body, headers=headers, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    return data["content"][0]["text"]


# ═══════════════════════════════════════════════════════════════════════════
#  UNIFIED AI ENGINE CLASS
# ═══════════════════════════════════════════════════════════════════════════

class AIEngine:
    """
    Multi-provider AI engine with automatic fallback.

    Usage:
        engine = AIEngine()
        engine.set_key("gemini", "your-api-key")

        # Text query
        response = engine.ask("What is the best fertilizer for rice?")

        # Vision query
        response = engine.ask_with_image(pil_image, "Identify the disease on this leaf")
    """

    def __init__(self):
        self._keys = {}
        self._chain = list(DEFAULT_CHAIN)
        self._last_provider = None
        self._last_error = None
        # Load keys from environment
        for pid, cfg in PROVIDERS.items():
            env_val = os.environ.get(cfg["env_key"], "")
            if env_val:
                self._keys[pid] = env_val

    def set_key(self, provider: str, key: str):
        """Set API key for a provider."""
        if provider in PROVIDERS and key and key.strip():
            self._keys[provider] = key.strip()

    def remove_key(self, provider: str):
        """Remove API key for a provider."""
        self._keys.pop(provider, None)

    def get_available_providers(self) -> list:
        """Return list of providers with API keys configured."""
        return [p for p in self._chain if p in self._keys]

    def get_provider_info(self) -> list:
        """Return info about all providers and their status."""
        result = []
        for pid in self._chain:
            cfg = PROVIDERS[pid]
            result.append({
                "id": pid,
                "name": cfg["name"],
                "has_key": pid in self._keys,
                "free_tier": cfg["free_tier"],
                "supports_vision": cfg["supports_vision"],
            })
        return result

    @property
    def last_provider(self) -> str:
        """Name of the provider that handled the last request."""
        return self._last_provider

    @property
    def last_error(self) -> str:
        """Error message from the last failed attempt."""
        return self._last_error

    def _call_provider(self, provider: str, prompt: str,
                       image_b64: str = None, system: str = None) -> str:
        """Call a specific provider."""
        api_key = self._keys.get(provider)
        if not api_key:
            raise ValueError(f"No API key for {provider}")

        if provider == "gemini":
            return _call_gemini(api_key, prompt, image_b64, system=system)
        elif provider == "claude":
            return _call_claude(api_key, prompt, image_b64, system=system)
        else:  # openai, groq, xai — all OpenAI-compatible
            return _call_openai_compatible(provider, api_key, prompt,
                                          image_b64, system=system)

    def ask(self, prompt: str, system: str = None,
            prefer_provider: str = None) -> str:
        """
        Send a text query through the fallback chain.
        Returns the AI response text, or raises RuntimeError if all fail.
        """
        return self._run_chain(prompt, image_b64=None, system=system,
                               need_vision=False, prefer_provider=prefer_provider)

    def ask_with_image(self, image: Image.Image, prompt: str,
                       system: str = None, prefer_provider: str = None) -> str:
        """
        Send an image + text query through the fallback chain (vision).
        Returns the AI response text, or raises RuntimeError if all fail.
        """
        image_b64 = _encode_image_base64(image)
        return self._run_chain(prompt, image_b64=image_b64, system=system,
                               need_vision=True, prefer_provider=prefer_provider)

    def ask_json(self, prompt: str, system: str = None,
                 prefer_provider: str = None) -> dict:
        """
        Like ask(), but expects JSON response. Parses and returns dict.
        Wraps the prompt to request JSON output.
        """
        json_prompt = (
            f"{prompt}\n\n"
            "IMPORTANT: Respond ONLY with valid JSON. No markdown, no code fences, "
            "no explanation before or after. Just the JSON object."
        )
        raw = self.ask(json_prompt, system=system, prefer_provider=prefer_provider)
        return _parse_json_response(raw)

    def ask_json_with_image(self, image: Image.Image, prompt: str,
                            system: str = None,
                            prefer_provider: str = None) -> dict:
        """Like ask_with_image(), but expects JSON response."""
        json_prompt = (
            f"{prompt}\n\n"
            "IMPORTANT: Respond ONLY with valid JSON. No markdown, no code fences, "
            "no explanation before or after. Just the JSON object."
        )
        image_b64 = _encode_image_base64(image)
        raw = self._run_chain(json_prompt, image_b64=image_b64, system=system,
                              need_vision=True, prefer_provider=prefer_provider)
        return _parse_json_response(raw)

    def _run_chain(self, prompt: str, image_b64: str = None,
                   system: str = None, need_vision: bool = False,
                   prefer_provider: str = None) -> str:
        """Run through the fallback chain until one provider succeeds."""
        errors = []
        chain = list(self._chain)

        # If a preferred provider is specified and available, try it first
        if prefer_provider and prefer_provider in self._keys:
            chain = [prefer_provider] + [p for p in chain if p != prefer_provider]

        for provider in chain:
            if provider not in self._keys:
                continue
            if need_vision and not PROVIDERS[provider]["supports_vision"]:
                continue
            try:
                result = self._call_provider(provider, prompt, image_b64,
                                             system=system)
                self._last_provider = PROVIDERS[provider]["name"]
                self._last_error = None
                return result
            except Exception as e:
                err_msg = f"{PROVIDERS[provider]['name']}: {str(e)[:200]}"
                errors.append(err_msg)
                continue

        self._last_provider = None
        self._last_error = "; ".join(errors) if errors else "No API keys configured"
        raise RuntimeError(
            f"All AI providers failed. Errors: {self._last_error}"
        )


# ═══════════════════════════════════════════════════════════════════════════
#  JSON RESPONSE PARSER
# ═══════════════════════════════════════════════════════════════════════════

def _parse_json_response(raw: str) -> dict:
    """Parse JSON from AI response, handling markdown code fences."""
    text = raw.strip()
    # Remove markdown code fences
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove first and last lines if they're fences
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try to find JSON object in the text
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            return json.loads(text[start:end])
        raise


# ═══════════════════════════════════════════════════════════════════════════
#  SYSTEM PROMPTS FOR AGRICULTURE
# ═══════════════════════════════════════════════════════════════════════════

SYSTEM_AGRI_EXPERT = """You are an expert agricultural scientist and farming advisor for Indian agriculture.
You have deep knowledge of crop management, soil science, plant pathology, entomology, and agricultural economics.
You give practical, actionable advice suited to Indian farming conditions.
Keep responses concise and well-structured. Use bullet points where appropriate."""

SYSTEM_DISEASE_DETECTOR = """You are an expert plant pathologist specializing in crop disease identification.
Analyze the provided crop/leaf image and identify:
1. The crop type visible in the image
2. Any disease symptoms (spots, discoloration, wilting, lesions, etc.)
3. The specific disease name if identifiable
4. Severity level (Healthy, Low, Moderate, High, Critical)
5. Treatment recommendations

Be specific and practical. If the image is unclear or not a crop, say so."""

SYSTEM_IMAGE_ANALYZER = """You are an expert agricultural remote sensing analyst.
Analyze the provided field/crop image and assess:
1. Crop type and growth stage
2. Overall crop health (score 0-100)
3. Signs of stress, disease, nutrient deficiency, or pest damage
4. Irrigation status
5. Weed presence
6. Estimated yield potential (Low/Medium/High)
Give practical observations that a farmer would find useful."""

SYSTEM_SOIL_INTERPRETER = """You are an expert soil scientist specializing in Indian agricultural soils.
Interpret the provided soil analysis data and explain:
1. What the numbers mean in plain language
2. Which nutrients are deficient or excess
3. Specific amendments recommended (with quantities)
4. Best crops for this soil profile
5. Long-term soil health improvement strategy
Keep it practical and understandable for farmers."""

SYSTEM_CHATBOT = """You are 'AgriBot', a friendly AI farming assistant for Indian agriculture.
You help farmers and agriculture students with:
- Crop selection and rotation advice
- Pest and disease identification
- Soil health management
- Weather-related farming decisions
- Market and pricing insights
- Government scheme information (PM-KISAN, Fasal Bima, etc.)
- Organic farming techniques

Be helpful, conversational, and practical. Use examples farmers can relate to.
If you don't know something, say so rather than guessing.
Keep replies concise — 2-4 paragraphs max unless asked for detail."""


# ═══════════════════════════════════════════════════════════════════════════
#  CONVENIENCE FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

# Global engine instance (created once, shared across the app)
_global_engine = None

def get_engine() -> AIEngine:
    """Get or create the global AI engine instance."""
    global _global_engine
    if _global_engine is None:
        _global_engine = AIEngine()
    return _global_engine


def ai_detect_disease(image: Image.Image, crop: str = None) -> dict:
    """
    Use AI vision to detect disease from a crop/leaf image.
    Returns dict compatible with disease_detector.predict() format.
    """
    engine = get_engine()
    crop_hint = f"The crop is {crop}. " if crop else ""

    prompt = f"""{crop_hint}Analyze this crop/leaf image for diseases.

Return a JSON object with these exact keys:
{{
    "disease": "disease name or Healthy",
    "confidence": 0.0 to 1.0,
    "severity": "None" or "Low" or "Moderate" or "High" or "Critical",
    "crop_detected": "crop name",
    "symptoms": ["list of observed symptoms"],
    "recommendations": ["list of treatment recommendations"],
    "description": "brief description of the condition"
}}"""

    try:
        result = engine.ask_json_with_image(
            image, prompt, system=SYSTEM_DISEASE_DETECTOR
        )
        # Ensure required keys exist
        result.setdefault("disease", "Unknown")
        result.setdefault("confidence", 0.5)
        result.setdefault("severity", "Unknown")
        result.setdefault("recommendations", [])
        result.setdefault("description", "")
        result["ai_provider"] = engine.last_provider
        return result
    except Exception as e:
        return {
            "disease": "Analysis unavailable",
            "confidence": 0.0,
            "severity": "Unknown",
            "recommendations": [f"AI analysis failed: {str(e)[:100]}. Using offline detection."],
            "description": "Could not reach AI service.",
            "ai_provider": None,
            "error": str(e)[:200]
        }


def ai_analyze_image(image: Image.Image) -> dict:
    """
    Use AI vision to analyze a crop/field image.
    Returns a rich analysis dict.
    """
    engine = get_engine()

    prompt = """Analyze this agricultural image. Return a JSON object with:
{
    "scene_type": "crop_field" or "soil" or "greenhouse" or "harvested" or "other",
    "crop_detected": "crop name or Unknown",
    "growth_stage": "seedling/vegetative/flowering/fruiting/mature",
    "health_score": 0 to 100,
    "health_label": "Excellent/Good/Fair/Poor/Critical",
    "observations": ["list of key observations"],
    "stress_signs": ["any stress indicators noticed"],
    "pest_disease_signs": ["any pest or disease signs"],
    "irrigation_status": "Adequate/Needs water/Waterlogged/Cannot determine",
    "weed_presence": "None/Low/Moderate/High",
    "yield_potential": "Low/Medium/High",
    "recommendations": ["actionable recommendations for the farmer"],
    "confidence": 0.0 to 1.0
}"""

    try:
        result = engine.ask_json_with_image(
            image, prompt, system=SYSTEM_IMAGE_ANALYZER
        )
        result["ai_provider"] = engine.last_provider
        return result
    except Exception as e:
        return {
            "error": str(e)[:200],
            "ai_provider": None
        }


def ai_soil_interpretation(soil_data: dict) -> str:
    """
    Use AI to interpret soil analysis results in plain language.
    soil_data should include: nitrogen, phosphorus, potassium, pH, etc.
    """
    engine = get_engine()

    prompt = f"""Interpret this soil analysis for an Indian farmer:

Soil Data:
- Nitrogen: {soil_data.get('nitrogen', 'N/A')} kg/ha
- Phosphorus: {soil_data.get('phosphorus', 'N/A')} kg/ha
- Potassium: {soil_data.get('potassium', 'N/A')} kg/ha
- pH: {soil_data.get('ph', 'N/A')}
- Organic Carbon: {soil_data.get('organic_carbon', 'N/A')}%
- Soil Type: {soil_data.get('soil_type', 'N/A')}
- Crop: {soil_data.get('crop', 'Not specified')}
- Overall Score: {soil_data.get('overall_score', 'N/A')}/100
- Deficiencies: {', '.join(soil_data.get('deficiencies', [])) or 'None detected'}

Provide:
1. Plain-language explanation of what these numbers mean
2. What the farmer should do to improve soil health
3. Which 3 crops would thrive best in this soil
4. Specific fertilizer recommendations (with quantities per acre)
5. Long-term soil improvement plan (1-2 sentences)

Keep it practical and concise — maximum 300 words."""

    try:
        return engine.ask(prompt, system=SYSTEM_SOIL_INTERPRETER)
    except Exception:
        return ""


def ai_smart_recommendations(crop: str, soil_data: dict = None,
                             weather_data: dict = None,
                             yield_prediction: float = None) -> str:
    """
    Use AI to generate context-aware farming recommendations.
    """
    engine = get_engine()

    context_parts = [f"Crop: {crop}"]
    if soil_data:
        context_parts.append(
            f"Soil: N={soil_data.get('nitrogen','?')}kg/ha, "
            f"P={soil_data.get('phosphorus','?')}kg/ha, "
            f"K={soil_data.get('potassium','?')}kg/ha, "
            f"pH={soil_data.get('ph','?')}, "
            f"Type={soil_data.get('soil_type','?')}"
        )
    if weather_data:
        context_parts.append(
            f"Weather: Temp={weather_data.get('temperature','?')}°C, "
            f"Humidity={weather_data.get('humidity','?')}%, "
            f"Rain={weather_data.get('precipitation','?')}mm"
        )
    if yield_prediction:
        context_parts.append(f"Predicted Yield: {yield_prediction:.2f} tonnes/ha")

    context = "\n".join(context_parts)

    prompt = f"""Given this farming context:
{context}

Provide smart, actionable farming recommendations covering:
1. **Immediate Actions** (what to do this week)
2. **Nutrient Management** (specific fertilizer advice)
3. **Pest & Disease Watch** (common threats for this crop + current weather)
4. **Water Management** (irrigation schedule based on weather)
5. **Yield Optimization** (tips to maximize output)

Be specific to Indian farming practices. Use quantities and timelines.
Keep total response under 400 words. Use markdown formatting."""

    try:
        return engine.ask(prompt, system=SYSTEM_AGRI_EXPERT)
    except Exception:
        return ""


def ai_chat(message: str, history: list = None) -> str:
    """
    Chat with the AI farming assistant.
    history: list of {"role": "user"/"assistant", "content": str}
    """
    engine = get_engine()

    # Build a self-contained prompt with history context
    if history and len(history) > 0:
        # Include last few exchanges for context
        recent = history[-6:]  # last 3 exchanges
        context = "Previous conversation:\n"
        for msg in recent:
            role = "Farmer" if msg["role"] == "user" else "AgriBot"
            context += f"{role}: {msg['content']}\n"
        prompt = f"{context}\nFarmer: {message}\n\nRespond as AgriBot:"
    else:
        prompt = message

    try:
        return engine.ask(prompt, system=SYSTEM_CHATBOT)
    except RuntimeError as e:
        return (
            "I'm sorry, I couldn't connect to any AI service right now. "
            "Please check your API keys in the sidebar settings.\n\n"
            f"Error: {str(e)[:200]}"
        )
