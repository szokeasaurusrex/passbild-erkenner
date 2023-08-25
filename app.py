from flask import Flask, render_template, request
from PIL import Image, UnidentifiedImageError
from passbild_erkenner import is_passbild, reason_message

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}

app = Flask(__name__)


def allowed_file(filename: str):
    return filename.split(".")[-1].lower() in ALLOWED_EXTENSIONS


@app.route("/")
def hello_world():
    return render_template("index.html")


@app.route("/submit", methods=["POST"])
def submit():
    if "photo" not in request.files or request.files["photo"].filename == "":
        return "No file uploaded!", 400

    if not allowed_file(request.files["photo"].filename):
        return (
            f"Incorrect file type. Only files with the following extensions are accepted: {', '.join(ALLOWED_EXTENSIONS)}.",
            415,
        )

    try:
        im = Image.open(request.files["photo"])
    except UnidentifiedImageError:
        return "Image format is not supported. Your image might be corrupted.", 415

    im_is_passbild, reason = is_passbild(im)

    if im_is_passbild:
        return "Das ist ein Passbild! 😀"

    return f"Das ist leider kein Passbild! 🙁\n{reason_message(reason)}"
