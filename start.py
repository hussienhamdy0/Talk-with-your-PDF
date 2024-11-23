# app.py
import subprocess
import os
def main():
    print("Starting the app!")
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    result = subprocess.run(["streamlit run app.py"], shell=True, capture_output=False, text=True)
    while True:
        command = input("Type 'exit' to quit or anything else to continue: ").lower()
        if command == 'exit':
            print("Exiting the app. Goodbye!")
            break
        else:
            print(f"You typed: {command}")

if __name__ == "__main__":
    main()
