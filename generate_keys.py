import pickle
import streamlit_authenticator as stauth
from pathlib import Path


names = ["Angelica Lim", "Loren Joe Alysa Buslon"]
usernames = ["Angelicalim", "Lorenbuslon"]
passwords = ["angelica123", "loren123"]

# Hash the passwords
hashed_passwords = stauth.Hasher(passwords).generate()

# Define file path
file_path = Path(__file__).parent / "hashed_pw.pkl"

# Save the hashed passwords to a pickle file
with open(file_path, "wb") as file:
    pickle.dump(hashed_passwords, file)
