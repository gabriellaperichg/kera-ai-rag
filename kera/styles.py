from pathlib import Path
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_styles():
    css_path = PROJECT_ROOT / "styles" / "style.css"
    css = css_path.read_text(encoding="utf-8")
    # O CSS precisa ser inserido como HTML para que o navegador aplique os estilos.
    st.markdown("<style>" + css + "</style>", unsafe_allow_html=True)
