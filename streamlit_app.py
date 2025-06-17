import streamlit as st
from sqlglot.dialects import DIALECTS
from sqlglot.llm_wrapper import LLMWrapper
from sqlglot import transpile
from sqlglot.errors import ParseError

st.title("SQL to SingleStore Translator")

sql = st.text_area("Enter SQL to convert")

source_dialects = [d.lower() for d in DIALECTS if d.lower() != "singlestore"]
source_dialects = sorted(source_dialects)
default_index = source_dialects.index("tsql") if "tsql" in source_dialects else 0
source = st.selectbox(
    "Input dialect",
    source_dialects,
    index=default_index,
)

# Add checkbox for stored procedure processing
is_stored_procedure = st.checkbox("Process as stored procedure (uses LLM)")

if st.button("Convert"):
    if sql:
        try:
            if is_stored_procedure:
                # For stored procedures, manually implement the flow to show each step
                wrapper = LLMWrapper()
                
                st.info("Step 1: Using LLM to decompose stored procedure into individual statements...")
                
                # Step 1: Decompose procedure into individual statements
                try:
                    statements = wrapper._decompose_procedure(sql)
                    st.subheader("Decomposed Statements")
                    for i, stmt in enumerate(statements, 1):
                        st.code(f"-- Statement {i}\n{stmt}", language="sql")
                    
                    st.info("Step 2: Transpiling each statement individually...")
                    
                    # Step 2: Transpile each statement
                    converted = []
                    for i, stmt in enumerate(statements, 1):
                        text = stmt.rstrip(";")
                        try:
                            transpiled = transpile(text, read=source, write="singlestore")[0]
                            converted.append(transpiled + ";")
                            st.write(f"**Statement {i} transpiled:**")
                            st.code(transpiled, language="sql")
                        except Exception as e:
                            st.warning(f"Statement {i} transpilation failed: {str(e)} - keeping original")
                            converted.append(text + ";")
                            st.code(text, language="sql")
                    
                    st.info("Step 3: Using LLM to reassemble into final SingleStore stored procedure...")
                    
                    # Step 3: Reassemble into final procedure
                    joined = "\n".join(converted)
                    try:
                        final_procedure = wrapper._reassemble_procedure(sql, joined)
                        st.subheader("Final SingleStore Stored Procedure")
                        st.code(final_procedure, language="sql")
                    except Exception as e:
                        st.error(f"Failed to reassemble procedure: {str(e)}")
                        st.subheader("Transpiled Statements (fallback)")
                        st.code(joined, language="sql")
                        
                except Exception as e:
                    st.error(f"Failed to decompose procedure: {str(e)}")
                    
            else:
                # For regular queries, just use SQLGlot transpilation
                transpiled = transpile(sql, read=source, write="singlestore")[0]
                st.subheader("SingleStore SQL")
                st.code(transpiled, language="sql")
        except ParseError as pe:
            st.error(f"SQL parsing error: {str(pe)}")
            st.info("💡 Tip: For complex stored procedures, try checking the 'Process as stored procedure' option which uses LLM to handle advanced syntax.")
        except ValueError as ve:
            st.error(str(ve))
        except Exception as e:
            st.error(f"Error: {str(e)}")
            st.info("💡 Tip: For complex stored procedures, try checking the 'Process as stored procedure' option.")
