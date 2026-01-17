import face_recognition
import numpy as np

def test_imports():
    print("--> Testing Imports...")
    # This checks if the compiled C++ extensions (dlib) loaded correctly
    assert face_recognition.api.dlib is not None
    print("    Imports Successful.")

def test_encoding():
    print("--> Testing Face Encoding (Math Check)...")
    # Create a dummy black image (100x100 pixels, RGB)
    dummy_image = np.zeros((100, 100, 3), dtype=np.uint8)
    
    # Try to locate faces (should find none, but shouldn't crash)
    locations = face_recognition.face_locations(dummy_image)
    print(f"    Pass 1: Ran face_locations without crash. Found {len(locations)} faces.")
    
    # Check if we can access the model
    try:
        # This triggers the heavy model loading
        face_recognition.face_encodings(dummy_image, locations)
        print("    Pass 2: Model loaded and encoding ran successfully.")
    except Exception as e:
        print(f"    FAILED during model execution: {e}")
        exit(1)

if __name__ == "__main__":
    try:
        test_imports()
        test_encoding()
        print("--- SMOKE TEST PASSED ---")
        exit(0)
    except Exception as e:
        print(f"--- SMOKE TEST FAILED: {e} ---")
        exit(1)