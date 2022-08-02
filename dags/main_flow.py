import json
import os
import re
import shutil
from datetime import datetime
from glob import glob
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import requests
import tensorflow as tf
from airflow import DAG
from airflow.models import Variable
from airflow.operators.python import PythonOperator
from airflow.providers.amazon.aws.hooks.s3 import S3Hook
from imutils import paths
from tensorflow.keras.applications.mobilenet import preprocess_input
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing import image

from functools import wraps
from time import time


# Constants
BASE = Variable.get("base")
CENTRAL = os.path.join(BASE, "dataset--central")
DETECTIONS = os.path.join(BASE, "output--detections")
DISCOVERY = os.path.join(BASE, "output--predictions")
DOWNLOADS = os.path.join(BASE, "downloads")
MAIN = os.path.join(BASE, "dataset--main")
MODEL = os.path.join(BASE, "model")
MODELS = Variable.get("models", deserialize_json=True)
UNKNOWN = os.path.join(BASE, "dataset--unknown")
GCS_ENDPOINT = Variable.get("gcs_endpoint", deserialize_json=True)


def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        print('func:%r took: %2.4f sec' % (f.__name__, te-ts))
        print('func:%r args:[%r, %r] took: %2.4f sec' % (f.__name__, args, kw, te-ts))
        return result
    return wrap



class Cropper:
    def __init__(self, dataset):
        self.new_dataset = dataset

    @staticmethod
    def preprocess_image(img: tf.Tensor):
        img = tf.image.decode_jpeg(img, channels=3)
        shapes = tf.shape(img)
        h, w = shapes[0], shapes[1]
        if h > w:
            cropped_image = tf.image.crop_to_bounding_box(img, (h - w) // 2, 0, w, w)
        else:
            cropped_image = tf.image.crop_to_bounding_box(img, 0, (w - h) // 2, h, h)

        x = tf.image.resize(cropped_image, (224, 224))
        x /= 255
        return x

    def load_and_preprocess_image(self, path: str):
        img = tf.io.read_file(path)
        return self.preprocess_image(img)

    def central_square(self, images):
        Path(self.new_dataset).mkdir(parents=True, exist_ok=True)
        for image_path in images:
            file_name = image_path.split(os.path.sep)[-1]
            folder = image_path.split(os.path.sep)[-2]
            folder_path = os.path.join(self.new_dataset, folder)
            Path(folder_path).mkdir(parents=True, exist_ok=True)
            target = os.path.join(folder_path, file_name)
            if not os.path.isfile(target):
                tf_cropped = self.load_and_preprocess_image(image_path)
                tf.keras.preprocessing.image.save_img(
                    target, tf_cropped, quality=100, subsampling=0
                )


def clear_environment():
    b = datetime.now().replace(microsecond=0)
    ignore = ('.py', '.txt', '.html')
    scan = glob(os.path.join(os.getcwd(), '**', '*.*'), recursive=True)
    trash = [f for f in scan if os.path.splitext(f)[1] not in ignore and os.path.isfile(f)]
    [os.remove(f) for f in trash]
    e = datetime.now().replace(microsecond=0)
    print(f'Cleaned up! -- {len(trash)} -- {e - b}')


def get_plant_name_maturity(file):
    regex = re.compile(r"\d+")
    exclude = ("train", "noProblem", "noNotes", "hasProblem", "hasNotes")
    filtered = [w for w in [i for i in file.split("-") if not regex.match(i)] if w not in exclude]
    plant_type = os.path.basename(os.path.dirname(file))
    age = filtered[-1].lower()
    return plant_type, age


def load(model):
    Path(MODEL).mkdir(parents=True, exist_ok=True)
    s3 = S3Hook()
    for f in ("classes.csv", "model.json", "weights.hdf5"):
        target = os.path.join(MODEL, model)
        Path(target).mkdir(parents=True, exist_ok=True)
        x = s3.download_file(
            bucket_name="sm--models",
            key=os.path.join(model, f),
            local_path=target,
        )
        shutil.move(x, x.replace(os.path.basename(x), f))

    model_base = os.path.join(MODEL, model)
    model_json = os.path.join(model_base, "model.json")
    weights = os.path.join(model_base, "weights.hdf5")
    # load json and create model
    with open(model_json, "r") as f:
        loaded_model_json = f.read()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(weights)

    classes = pd.read_csv(os.path.join(model_base, "classes.csv"), index_col=0).label.to_list()
    return loaded_model, classes


def prepare_image(file):
    img = image.load_img(file, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array_expanded_dims = np.expand_dims(img_array, axis=0)
    return preprocess_input(img_array_expanded_dims)

@timing
def _select_exist_files(**context):
    # Create Temp Folders
    dataset_main = context.get("main")
    downloads = context.get("downloads")
    unknown_folder = context.get("unknown")
    Path(dataset_main).mkdir(parents=True, exist_ok=True)
    Path(downloads).mkdir(parents=True, exist_ok=True)
    Path(unknown_folder).mkdir(parents=True, exist_ok=True)

    print(f"Free: {shutil.disk_usage(os.path.dirname(dataset_main)).free / 1048576} Mb")
    s3 = S3Hook()
    uploaded = s3.list_keys(os.path.basename(dataset_main))

    """
    Creation of a plant dictionary, as it was observed that in the construction of the
    image file name there is a species identification code. Then, using this code, we
    were able to reveal files that are "unknown" by renaming them through the code
    and species dictionary.
    """
    plants = []
    for data in uploaded:
        try:
            plant_id = os.path.basename(data).split("-")[1]
        except (ValueError, IndexError):
            continue
        plant_name = os.path.basename(os.path.dirname(data))
        plants.append((plant_id, plant_name))

    print(f"example data: {data}")
    print(f"example plant_name: {plant_name}")

    from_to = list(set(plants))
    from_to.sort(key=lambda x: x[0])
    from_to = [{"id": key, "name": val} for key, val in from_to]
    context["ti"].xcom_push(key="dict_plants", value=from_to)

    all_images = [
        os.path.splitext(i.split("/")[-1])[0]
        for i in uploaded
        if os.path.splitext(i.split("/")[-1])[0] != ""
    ]
    print(f"Example all_images: {all_images[-1]}")
    unknown_images = [
        os.path.splitext(i.split(os.path.sep)[-1])[0]
        for i in s3.list_keys(os.path.basename(unknown_folder))
        if os.path.splitext(i.split("/")[-1])[0] != ""
    ]

    print(f"Example unknown_images: {unknown_images[-1]}")

    all_images.extend(unknown_images)
    context["ti"].xcom_push(key="existing", value=all_images)
    print(f"select_exist_files has {len(all_images)} images")

@timing
def _check_updates(**context):
    responses = []
    last_ts = 0
    while True:
        response = requests.get(context.get("uri").format(ts=last_ts))
        lines = response.json()
        try:
            result = lines[1:]
            responses.extend(result)
            if len(result) < 1000:  # pagination
                break
            last_ts = responses[-1][1]
        except ValueError:
            break
    context["ti"].xcom_push(key="responses", value=responses)
    print(f"check_updates has {len(responses)}")


def _scan_images(**context):
    clear_environment()

    downloads = context.get("downloads")
    source = context.get("ti").xcom_pull(key="responses")
    dict_plants = context.get("ti").xcom_pull(key="dict_plants")
    existing = context.get("ti").xcom_pull(key="existing")

    n_existing = 0
    n_new = 0
    i = 0

    for line in source:
        i+=1
        plant_name = line[2].lower() if line[2] else "unknown"
        image_url = line[4]
        file_name = os.path.basename(image_url)
        plant_id = file_name.split("-")[1]
        if "null" in file_name:
            if plant_name != "unknown":
                new_file_name = file_name.replace("null", plant_name)
            else:
                try:
                    plant_name = list(filter(lambda item: item["id"] == plant_id, dict_plants))[0].get(
                        "name"
                    )
                    new_file_name = file_name.replace("null", plant_name)
                except IndexError:
                    new_file_name = file_name
        else:
            new_file_name = file_name

        if os.path.splitext(new_file_name)[0] in existing:
            n_existing += 1
            continue

        target = os.path.join(downloads, plant_name)
        Path(target).mkdir(parents=True, exist_ok=True)
        new_path = os.path.join(target, new_file_name)
        if not os.path.isfile(new_path):
            r = requests.get(image_url, stream=True)
            if r.status_code == 200:
                open(new_path, "wb").write(r.content)
                n_new += 1
    print(f"Existing images = {n_existing}")
    print(f"New images      = {n_new}")
    print(f"Total images    = {i}")


def _organize(**context):
    dataset_main = context.get("main")
    downloads = context.get("downloads")
    unknown = context.get("unknown")

    updates = list(paths.list_images(downloads))
    d = 0
    waste = 0
    for update in updates:
        label = update.split(os.path.sep)[-2]
        file_name = update.split(os.path.sep)[-1]
        target = unknown if label == "unknown" else os.path.join(dataset_main, label)
        if os.stat(update).st_size > 1024 * 140:  # 140 kb
            if not os.path.isfile(os.path.join(target, file_name).replace("png", "jpg")):
                Path(target).mkdir(parents=True, exist_ok=True)
                try:
                    cv2.imwrite(
                        os.path.join(target, file_name).replace("png", "jpg"),
                        cv2.imread(update),
                        [int(cv2.IMWRITE_JPEG_QUALITY), 90],
                    )
                    d += 1
                except Exception as err:
                    print(err, update)
        else:
            waste += 1

    shutil.rmtree(downloads)
    print(f"{d} new images!")
    print(f"{waste} trashed in!")


def _crop_images(**context):
    new_images = list(paths.list_images(context.get("main")))
    Cropper(context.get("central")).central_square(new_images)


def _predict_images(**context):
    central = context.get("central")
    discovery = context.get("discovery")
    dataset = list(paths.list_images(central))
    model_maturity, labels_maturity = load(context.get("model_maturity"))
    model_plant, labels_plant = load(context.get("model_plant"))

    output = []
    for u in dataset:
        actual_plant_type, actual_maturity = get_plant_name_maturity(u)
        preprocessed_image = prepare_image(u)
        obj_key = os.path.sep.join(u.split(os.path.sep)[-2:])
        image_id = os.path.splitext(os.path.basename(obj_key))[0].split("-")[-1]
        s3_path = f"https://{os.path.basename(central)}.s3.amazonaws.com/images/{obj_key}"

        plant_prediction = model_plant.predict(preprocessed_image)[0]
        top_3_plant_predictions = [
            {
                "class": labels_plant[t],
                "confidence": round(plant_prediction[t] * 100, 2),
            }
            for t in plant_prediction.argsort()[-3:][::-1]
        ]

        maturity_prediction = model_maturity.predict(preprocessed_image)[0]
        top_3_maturity_predictions = [
            {
                "class": labels_maturity[t],
                "confidence": round(maturity_prediction[t] * 100, 2),
            }
            for t in maturity_prediction.argsort()[-3:][::-1]
        ]

        response = {
            "path": s3_path,
            "image_id": image_id,
            "trained_class": True if actual_plant_type in labels_plant else False,
            "plant_type": {
                "actual": actual_plant_type,
                "predictions": top_3_plant_predictions,
            },
            "maturity": {
                "actual": actual_maturity,
                "predictions": top_3_maturity_predictions,
            },
        }
        output.append(response)

    if len(output) > 0:
        Path(discovery).mkdir(parents=True, exist_ok=True)
        output_file = os.path.join(discovery, f'internal_{datetime.now().strftime("%Y-%m-%d")}.json')
        with open(output_file, "w") as f:
            json.dump(output, f, indent=4)
        print("Output file:", len(output))


def _upload_to_s3(**context):
    buckets = context.get("buckets")
    for bucket in buckets:
        new_uploaded = 0
        work_dir = list(paths.list_files(bucket, validExts=(".jpg", ".jpeg", ".png", ".json", ".csv")))

        bucket_name = os.path.basename(bucket)
        prefix = "" if bucket in (DISCOVERY, DETECTIONS) else "images"
        s3 = S3Hook(extra_args={'ContentType': 'image/jpeg'}) if prefix else S3Hook()
        uploaded = s3.list_keys(bucket_name, prefix=prefix)
        for img in work_dir:
            path = (
                os.path.basename(img)
                if bucket in (DISCOVERY, DETECTIONS)
                else os.path.join(prefix, os.path.sep.join(img.split(os.path.sep)[-2:]))
            )
            if path not in uploaded:
                s3.load_file(
                    bucket_name=bucket_name,
                    filename=img,
                    key=path,
                    replace=True,
                    acl_policy="public-read",
                )
                new_uploaded += 1

        print(f"New Uploaded: {new_uploaded} to {bucket_name}")


default_args = {
    "owner": "ML",
    "start_date": datetime(2021, 12, 1),
    "provide_context": True,
}
dag = DAG(
    dag_id="main_flow",
    description="Collect Internal Images",
    schedule_interval="@daily",
    default_args=default_args,
    tags=["etl"],
)

with dag:
    dag.doc_md = __doc__
    dag.doc_md = """
            #### Flow to check updates and download new images from the internal set.
        """
    select_exist_files = PythonOperator(
        task_id="select_exist_files",
        python_callable=_select_exist_files,
        op_kwargs={
            "main": MAIN,
            "unknown": UNKNOWN,
            "downloads": DOWNLOADS,
        },
    )

    check_updates = PythonOperator(
        task_id="check_updates",
        python_callable=_check_updates,
        op_kwargs={
            "uri": GCS_ENDPOINT.get("internal")
        },
    )

    scan_images = PythonOperator(
        task_id="scan_images",
        python_callable=_scan_images,
        op_kwargs={"downloads": DOWNLOADS},
    )

    organize = PythonOperator(
        task_id="organize",
        python_callable=_organize,
        op_kwargs={
            "main": MAIN,
            "unknown": UNKNOWN,
            "downloads": DOWNLOADS,
        },
    )

    crop_images = PythonOperator(
        task_id="crop_images",
        python_callable=_crop_images,
        op_kwargs={
            "main": MAIN,
            "central": CENTRAL,
        },
    )

    predict_images = PythonOperator(
        task_id="predict_images",
        python_callable=_predict_images,
        op_kwargs={
            "central": CENTRAL,
            "discovery": DISCOVERY,
            "model_maturity": MODELS.get("maturity"),
            "model_plant": MODELS.get("plant"),
        },
    )

    upload_to_s3 = PythonOperator(
        task_id="upload_to_s3",
        python_callable=_upload_to_s3,
        op_kwargs={"buckets": [MAIN, CENTRAL, UNKNOWN, DISCOVERY, DETECTIONS]},
    )

    [select_exist_files, check_updates] >> scan_images
    scan_images >> organize
    organize >> crop_images
    crop_images >> predict_images
    predict_images >> upload_to_s3


