import io
import contextlib

def run_sandbox(code: str, given_data) -> dict: #utility function
    buffer = io.StringIO() #empty fake in-memory
    try:
        with contextlib.redirect_stdout(buffer): #whatever is printed inside here goes into the memory
            exec(code, {"given_data": given_data})
            captured_text = buffer.getvalue()   #pulls everything from buffer to string
            return {"success": True, "stdout": captured_text, "error": None}
    except Exception as e:
        return {"success": False, "stdout": buffer.getvalue(), "error": str(e)}