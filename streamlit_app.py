import streamlit as st
from sqlglot.dialects import DIALECTS
from sqlglot.llm_wrapper import LLMWrapper

st.title("SQL to SingleStore Translator")

sql = st.text_area("Enter SQL to convert")

wrapper = LLMWrapper()

source_dialects = [d.lower() for d in DIALECTS if d.lower() != "singlestore"]
source_dialects = sorted(source_dialects)
default_index = source_dialects.index("tsql") if "tsql" in source_dialects else 0
source = st.selectbox(
    "Input dialect",
    source_dialects,
    index=default_index,
)

if st.button("Convert"):
    if sql:
        try:
            translated = wrapper.to_singlestore(sql, source)
            st.subheader("SingleStore SQL")
            st.code(translated, language="sql")
        except ValueError as ve:
            st.error(str(ve))
        except Exception as e:
            st.error(str(e))
