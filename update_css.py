import sys

with open('app.py', 'r', encoding='utf-8') as f:
    content = f.read()

start_marker = 'def _app_css():\n    return """'
end_marker = 'st.markdown(_app_css(), unsafe_allow_html=True)'

start_idx = content.find(start_marker)
end_idx = content.find(end_marker)

if start_idx == -1 or end_idx == -1:
    print('Could not find markers')
    sys.exit(1)

new_css = """def _app_css():
    return \"\"\"

<style>

@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;500;700;900&family=Rajdhani:wght@400;500;600;700&display=swap');

/* ══════════════════════════════════════════════════════════════════

   ROOT – Dark Cyber-Agriculture (Out of this World)

   ══════════════════════════════════════════════════════════════════ */

@keyframes cyberFloat {
    0%   { transform: translateY(0px) rotate(0deg); opacity: 0.1; }
    50%  { opacity: 0.8; }
    100% { transform: translateY(-100px) rotate(45deg); opacity: 0.1; }
}

@keyframes holographicPulse {
    0%   { opacity: 0.8; filter: drop-shadow(0 0 2px #00E676); }
    50%  { opacity: 1; filter: drop-shadow(0 0 10px #00E5FF); }
    100% { opacity: 0.8; filter: drop-shadow(0 0 2px #00E676); }
}

@keyframes nebulaDrift {
    0% { background-position: 0% 0%; }
    50% { background-position: 100% 100%; }
    100% { background-position: 0% 0%; }
}

[data-testid="stAppViewContainer"] {

    font-family: 'Rajdhani', sans-serif;

    color: #E0F2F1; /* Ice white / pale teal for text */

    background:

        /* ── Deep Space Nebula Base ── */
        radial-gradient(ellipse at 10% 20%, rgba(26, 8, 38, 0.8) 0%, transparent 50%),
        radial-gradient(ellipse at 90% 80%, rgba(4, 26, 30, 0.8) 0%, transparent 50%),
        radial-gradient(circle at 50% 50%, rgba(0, 30, 20, 0.6) 0%, transparent 60%),
        
        /* ── Cyber Grid ── */
        linear-gradient(rgba(0, 230, 118, 0.03) 1px, transparent 1px),
        linear-gradient(90deg, rgba(0, 230, 118, 0.03) 1px, transparent 1px),

        /* ── Void Black Floor ── */
        linear-gradient(180deg,
            #02040A 0%,      /* Deep void black */
            #050A14 40%,     /* Extremely dark blue/cyan */
            #030F12 80%,     /* Dark teal space */
            #00150F 100%     /* Deep dark green underbelly */
        );
        
    background-size: 100% 100%, 100% 100%, 100% 100%, 40px 40px, 40px 40px, 100% 100%;
    animation: nebulaDrift 60s ease infinite;

    background-attachment: fixed;

}

/* ── Floating Digital Spores (Cyber Particles) ── */
[data-testid="stAppViewContainer"]::before {
    content: "";
    position: fixed;
    top: 0; left: 0; right: 0; bottom: 0;
    pointer-events: none;
    z-index: 0;
    background:
        radial-gradient(1.5px 1.5px at 20% 80%, #00E676, transparent),
        radial-gradient(2px 2px at 60% 40%, #00E5FF, transparent),
        radial-gradient(1px 1px at 80% 90%, #B388FF, transparent),
        radial-gradient(2.5px 2.5px at 30% 20%, #64FFDA, transparent),
        radial-gradient(1px 1px at 10% 10%, #00E676, transparent),
        radial-gradient(2px 2px at 90% 30%, #00E5FF, transparent),
        radial-gradient(1.5px 1.5px at 50% 70%, #64FFDA, transparent);
    animation: cyberFloat 15s linear infinite;
    opacity: 0.6;
}

/* ── Holographic Base Glow at Bottom ── */

[data-testid="stAppViewContainer"]::after {

    content: "";

    position: fixed;

    bottom: 0; left: 0; right: 0;

    height: 100px;

    pointer-events: none;

    z-index: 0;

    background:
        linear-gradient(0deg, rgba(0, 230, 118, 0.15) 0%, rgba(0, 229, 255, 0.05) 50%, transparent 100%);

    border-top: 1px solid rgba(0, 230, 118, 0.3);
    box-shadow: 0 -5px 20px rgba(0, 230, 118, 0.2);

    animation: holographicPulse 4s ease-in-out infinite alternate;

}

/* ── Sidebar – Dark Glass Terminal ── */

[data-testid="stSidebar"] {

    background:

        linear-gradient(180deg, rgba(5, 10, 20, 0.85) 0%, rgba(2, 5, 10, 0.95) 100%);

    border-right: 1px solid rgba(0, 229, 255, 0.3);

    box-shadow: 5px 0 25px rgba(0, 229, 255, 0.1);

    backdrop-filter: blur(15px);

}

[data-testid="stSidebar"] * { color: #84FFFF !important; font-family: 'Rajdhani', sans-serif; }

/* ══════════════════════════════════════════════════════════════════

   HERO BANNER – Holographic Control Deck

   ══════════════════════════════════════════════════════════════════ */

.hero {

    background:

        linear-gradient(135deg, rgba(0, 20, 20, 0.6), rgba(5, 10, 25, 0.8));

    border: 1px solid rgba(0, 230, 118, 0.5);

    border-radius: 12px;

    padding: 36px 40px;

    margin-bottom: 28px;

    backdrop-filter: blur(20px);

    box-shadow: 
        0 15px 35px rgba(0, 0, 0, 0.8),
        inset 0 0 20px rgba(0, 230, 118, 0.1),
        0 0 10px rgba(0, 230, 118, 0.2);

    position: relative;

    overflow: hidden;

}

/* Futuristic Scanline over Hero */
.hero::after {
    content: "";
    position: absolute;
    top: 0; left: 0; right: 0; bottom: 0;
    pointer-events: none;
    background: repeating-linear-gradient(
        0deg,
        transparent,
        transparent 2px,
        rgba(0, 230, 118, 0.05) 2px,
        rgba(0, 230, 118, 0.05) 4px
    );
    opacity: 0.5;
}

.hero::before {

    content: "⚡🧬🛰️🌌";

    position: absolute;

    top: 20px; right: 20px;

    font-size: 2.5rem;

    opacity: 0.3;

    letter-spacing: 5px;
    filter: drop-shadow(0 0 10px #00E5FF);

}

.hero h1 {

    margin: 0; font-size: 3rem; font-weight: 900; font-family: 'Orbitron', sans-serif;

    background: linear-gradient(90deg, #00E676, #00E5FF, #B388FF);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: 2px;
    text-transform: uppercase;
    text-shadow: 0 0 20px rgba(0, 230, 118, 0.5);

}

.hero p {

    margin: 10px 0 0; font-size: 1.2rem; color: #B2DFDB;

    font-weight: 500;
    letter-spacing: 1px;

}

/* ══════════════════════════════════════════════════════════════════

   CARDS – Holographic Dark Glass

   ══════════════════════════════════════════════════════════════════ */

.card {

    background: rgba(10, 15, 25, 0.6);

    border: 1px solid rgba(0, 229, 255, 0.3);

    border-radius: 12px;

    padding: 24px;

    margin-bottom: 20px;

    backdrop-filter: blur(12px);

    box-shadow:

        0 10px 30px rgba(0, 0, 0, 0.8),
        inset 0 0 15px rgba(0, 229, 255, 0.05);

    transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);

}

.card:hover {

    transform: translateY(-5px);
    border-color: rgba(0, 230, 118, 0.8);

    box-shadow:

        0 15px 40px rgba(0, 0, 0, 0.9),
        0 0 20px rgba(0, 230, 118, 0.3),
        inset 0 0 20px rgba(0, 230, 118, 0.1);

}

.metric-card {

    background: linear-gradient(145deg, rgba(5, 20, 25, 0.8) 0%, rgba(2, 10, 15, 0.9) 100%);

    border: 1px solid rgba(0, 230, 118, 0.4);

    border-radius: 12px;

    padding: 22px 18px;

    text-align: center;
    
    box-shadow: 0 10px 20px rgba(0,0,0,0.8), inset 0 0 10px rgba(0, 230, 118, 0.1);

    transition: all 0.3s cubic-bezier(0.34, 1.56, 0.64, 1);
    position: relative;
    overflow: hidden;

}

.metric-card::after {
    content: ''; position: absolute; top: 0; left: 0; width: 100%; height: 2px;
    background: linear-gradient(90deg, transparent, #00E676, transparent);
}

.metric-card:hover { 
    transform: translateY(-8px) scale(1.02); 
    border-color: #00E5FF;
    box-shadow: 0 15px 30px rgba(0,0,0,0.9), 0 0 25px rgba(0, 229, 255, 0.3), inset 0 0 15px rgba(0, 229, 255, 0.1);
}

.metric-value { font-family: 'Orbitron', sans-serif; font-size: 2.2rem; font-weight: 700; color: #FFFFFF; letter-spacing: 2px; text-shadow: 0 0 15px #00E676, 0 0 5px #00E676; }

.metric-label { font-size: 0.9rem; color: #1DE9B6; margin-top: 8px; font-weight: 600; text-transform: uppercase; letter-spacing: 3px;}

/* ── Big result box – Reactor Core ── */

@keyframes reactorPulse {
    0% { box-shadow: 0 0 30px rgba(0, 230, 118, 0.2), inset 0 0 20px rgba(0, 230, 118, 0.1); border-color: rgba(0, 230, 118, 0.5); }
    100% { box-shadow: 0 0 60px rgba(0, 230, 118, 0.5), inset 0 0 40px rgba(0, 230, 118, 0.3); border-color: rgba(0, 230, 118, 1); }
}

.result-box {

    background:

        radial-gradient(circle at 50% 50%, rgba(0, 230, 118, 0.1) 0%, transparent 70%),

        linear-gradient(135deg, #02100A, #010508);

    border-radius: 16px;

    padding: 45px;

    text-align: center;

    border: 2px solid #00E676;

    animation: reactorPulse 2s ease-in-out infinite alternate;
    position: relative;
}

.result-box::before {
    content: '[ SYS.YIELD_CALC // OPTIMAL ]';
    position: absolute; top: 10px; left: 15px; font-family: 'Orbitron', sans-serif; font-size: 0.6rem; color: #00E676; letter-spacing: 2px; opacity: 0.7;
}

.result-value {

    font-family: 'Orbitron', sans-serif; font-size: 4.5rem; font-weight: 900; 
    
    color: #FFFFFF;
    text-shadow: 0 0 20px #00E676, 0 0 40px #00E676, 0 0 10px #FFFFFF;
    letter-spacing: 2px;
}

.result-label { font-size: 1.2rem; color: #64FFDA; margin-top: 15px; font-weight: 700; text-transform: uppercase; letter-spacing: 5px;}

/* ── Model card – Data Node ── */

.model-card {

    background: rgba(15, 10, 25, 0.6);

    border: 1px solid rgba(179, 136, 255, 0.4);

    border-left: 4px solid #B388FF;

    border-radius: 8px;

    padding: 18px 20px;

    margin-bottom: 12px;

    box-shadow: 0 8px 20px rgba(0,0,0,0.6);

    transition: all 0.3s ease;

}

.model-card:hover {

    border-color: #D500F9;
    border-left-color: #D500F9;

    transform: translateX(10px);

    background: rgba(25, 10, 40, 0.8);

    box-shadow: 0 15px 30px rgba(0,0,0,0.8), 0 0 15px rgba(213, 0, 249, 0.3);

}

/* ── Weather card – Atmospheric Scanner ── */

.weather-card {

    background:

        linear-gradient(135deg, rgba(2, 20, 30, 0.8), rgba(0, 10, 20, 0.9));

    border: 1px solid rgba(0, 229, 255, 0.4);

    border-radius: 12px;

    padding: 24px;

    text-align: center;

    box-shadow: 0 12px 24px rgba(0, 0, 0, 0.8), inset 0 0 15px rgba(0, 229, 255, 0.1);

}

.weather-value { font-family: 'Orbitron', sans-serif; font-size: 2.2rem; font-weight: 700; color: #FFFFFF; text-shadow: 0 0 15px #00E5FF; }

.weather-label { font-size: 0.9rem; color: #84FFFF; margin-top: 8px; font-weight: 700; text-transform: uppercase; letter-spacing: 2px;}

/* ── Image scan card ── */

.scan-card {

    background: linear-gradient(135deg, rgba(20, 5, 30, 0.8), rgba(10, 2, 20, 0.9));

    border: 1px solid rgba(213, 0, 249, 0.5);

    border-radius: 12px;

    padding: 24px;

    box-shadow: 0 15px 35px rgba(0,0,0,0.8), inset 0 0 20px rgba(213, 0, 249, 0.1);

}

/* ══════════════════════════════════════════════════════════════════

   SECTION HEADERS – Cyber typography

   ══════════════════════════════════════════════════════════════════ */

.section-header {

    font-family: 'Orbitron', sans-serif;
    font-size: 1.6rem; font-weight: 700; color: #FFFFFF;

    padding-bottom: 12px;

    border-bottom: 1px solid rgba(0, 229, 255, 0.3);

    margin-bottom: 28px;
    letter-spacing: 2px;
    text-transform: uppercase;
    position: relative;
    text-shadow: 0 0 10px rgba(0, 229, 255, 0.5);

}

.section-header::after {

    content: "";

    position: absolute; bottom: -1px; left: 0; width: 100px; height: 2px;

    background: #00E5FF;
    box-shadow: 0 0 10px #00E5FF;

}

.sub-header {

    font-family: 'Orbitron', sans-serif;
    font-size: 1.1rem; font-weight: 600; color: #00E676; margin-bottom: 16px;
    letter-spacing: 1px;
    text-transform: uppercase;

}

/* ══════════════════════════════════════════════════════════════════

   BUTTONS – UI Controls (Angular, Glowing)

   ══════════════════════════════════════════════════════════════════ */

[data-testid="stAppViewContainer"] div.stButton > button {

    background: rgba(0, 230, 118, 0.1) !important;

    color: #00E676 !important;

    border: 1px solid #00E676 !important;

    border-radius: 4px; /* Sharp corners */

    font-family: 'Orbitron', sans-serif;
    font-weight: 700;

    padding: 12px 28px;

    font-size: 1rem;

    width: 100%;

    text-transform: uppercase;
    letter-spacing: 2px;

    box-shadow: 0 0 10px rgba(0, 230, 118, 0.2), inset 0 0 10px rgba(0, 230, 118, 0.1);

    transition: all 0.2s ease;

}

[data-testid="stAppViewContainer"] div.stButton > button:hover {

    background: rgba(0, 230, 118, 0.2) !important;

    color: #FFFFFF !important;

    border-color: #FFFFFF !important;

    transform: translateY(-2px);

    box-shadow: 0 0 20px rgba(0, 230, 118, 0.6), inset 0 0 20px rgba(0, 230, 118, 0.3);

}

[data-testid="stAppViewContainer"] div.stButton > button:active {

    transform: translateY(1px);

    box-shadow: 0 0 5px rgba(0, 230, 118, 0.5);

}

/* ── Sidebar buttons ── */

[data-testid="stSidebar"] div.stButton > button {

    background: transparent !important;

    border: 1px solid transparent !important;
    border-left: 2px solid rgba(0, 229, 255, 0.3) !important;

    border-radius: 0 !important;

    color: #84FFFF !important;

    font-weight: 600 !important;

    font-size: 1rem !important;

    padding: 10px 15px !important;

    text-align: left !important;

    margin-bottom: 8px !important;

    box-shadow: none !important;

    transition: all 0.2s ease !important;
    text-transform: uppercase;
    letter-spacing: 1px;

}

[data-testid="stSidebar"] div.stButton > button:hover {

    background: linear-gradient(90deg, rgba(0, 229, 255, 0.1), transparent) !important;

    color: #FFFFFF !important;
    border-left: 2px solid #00E5FF !important;
    
    transform: translateX(5px) !important;
    text-shadow: 0 0 8px #00E5FF;
}

/* ── Active nav (primary) ── */

[data-testid="stSidebar"] div.stButton > button[kind="primary"],

[data-testid="stSidebar"] div.stButton > button[data-testid="stBaseButton-primary"] {

    background: linear-gradient(90deg, rgba(0, 230, 118, 0.2), transparent) !important;

    border: 1px solid rgba(0, 230, 118, 0.3) !important;
    border-left: 4px solid #00E676 !important;

    color: #FFFFFF !important;

    font-weight: 700 !important;
    
    box-shadow: inset 10px 0 20px rgba(0, 230, 118, 0.1) !important;
    text-shadow: 0 0 10px #00E676 !important;

}

[data-testid="stSidebar"] div.stButton > button[kind="primary"]:hover,

[data-testid="stSidebar"] div.stButton > button[data-testid="stBaseButton-primary"]:hover {

    background: linear-gradient(90deg, rgba(0, 230, 118, 0.3), transparent) !important;

    transform: translateX(5px) !important;
    border-color: #00E676 !important;

}

/* ── Retrain button ── */

[data-testid="stSidebar"] button[key*="retrain"] {

    background: rgba(213, 0, 249, 0.1) !important;

    border: 1px solid #D500F9 !important;
    border-left: 4px solid #D500F9 !important;

    color: #E040FB !important;

    box-shadow: 0 0 15px rgba(213, 0, 249, 0.2) !important;
    
    border-radius: 0 !important;

}
[data-testid="stSidebar"] button[key*="retrain"]:hover {
    box-shadow: 0 0 25px rgba(213, 0, 249, 0.5) !important;
    background: rgba(213, 0, 249, 0.2) !important;
    color: #FFF !important;
}

/* ══════════════════════════════════════════════════════════════════

   FORM ELEMENTS – Data input fields

   ══════════════════════════════════════════════════════════════════ */

div[data-baseweb="select"] > div,

div[data-baseweb="input"] > div,

textarea {

    background: rgba(2, 8, 15, 0.8) !important;

    border: 1px solid rgba(0, 229, 255, 0.3) !important;

    color: #FFFFFF !important;

    border-radius: 4px !important; 

    box-shadow: inset 0 0 10px rgba(0,0,0,0.8) !important;
    
    transition: all 0.2s ease !important;
    font-family: 'Rajdhani', sans-serif !important;
    font-size: 1.1rem !important;

}

div[data-baseweb="select"] > div:hover,
div[data-baseweb="input"] > div:focus-within {
    border-color: #00E5FF !important;
    box-shadow: 0 0 15px rgba(0, 229, 255, 0.2), inset 0 0 10px rgba(0,0,0,0.8) !important;
}

label { color: #84FFFF !important; font-weight: 600 !important; letter-spacing: 1px; font-size: 0.95rem !important; text-transform: uppercase;}

/* ── Sliders – Neon Bar ── */

div[data-baseweb="slider"] div[role="slider"] {

    background: #00E676 !important;

    border: 2px solid #FFF !important;

    box-shadow: 0 0 10px #00E676, 0 0 5px #FFF !important;
    border-radius: 0 !important;

}

/* ── Tabs ── */

.stTabs [data-baseweb="tab-list"] { gap: 10px; border-bottom: 1px solid rgba(0, 229, 255, 0.3); padding-bottom: 0px;}

.stTabs [data-baseweb="tab"] {

    background: transparent;

    border-radius: 4px 4px 0 0;

    color: #00E5FF;
    opacity: 0.6;

    font-family: 'Orbitron', sans-serif;
    font-weight: 600;

    padding: 12px 20px;
    
    transition: all 0.2s ease;
    text-transform: uppercase;
    letter-spacing: 1px;

}

.stTabs [data-baseweb="tab"]:hover {
    color: #FFFFFF;
    opacity: 1;
    background: rgba(0, 229, 255, 0.1);
}

.stTabs [aria-selected="true"] {

    background: rgba(0, 229, 255, 0.15) !important;

    color: #FFFFFF !important;
    opacity: 1 !important;

    border-bottom: 3px solid #00E5FF;
    border-top: 1px solid rgba(0, 229, 255, 0.5);
    border-left: 1px solid rgba(0, 229, 255, 0.5);
    border-right: 1px solid rgba(0, 229, 255, 0.5);
    
    box-shadow: inset 0 10px 20px rgba(0, 229, 255, 0.1) !important;
    text-shadow: 0 0 10px rgba(0, 229, 255, 0.8) !important;

}

/* ══════════════════════════════════════════════════════════════════

   WEATHER DISPLAY – inline in predict form

   ══════════════════════════════════════════════════════════════════ */

.weather-inline {

    background: rgba(2, 10, 20, 0.8);

    border: 1px solid rgba(0, 229, 255, 0.4);
    border-left: 4px solid #00E5FF;

    border-radius: 4px;

    padding: 15px 20px;

    margin-top: 15px;

    font-size: 1rem;
    
    font-weight: 600;
    letter-spacing: 1px;

    color: #B2DFDB;

    box-shadow: 0 5px 15px rgba(0, 0, 0, 0.5);

}

/* ── Hide default branding ── */

#MainMenu, footer, header { visibility: hidden; }
html { scroll-behavior: smooth; }

</style>

\"\"\"
"""

new_content = content[:start_idx] + new_css + content[end_idx:]

with open('app.py', 'w', encoding='utf-8') as f:
    f.write(new_content)

print('Successfully replaced _app_css')
