import logging
import random
from datetime import datetime
from PIL import Image
import numpy as np

from ActiveLearning.s3_helper import S3Ref, download_bytesio

AUTOANNOTATION_THRESHOLD = 0.50
JOB_TYPE = "groundtruth/object-detection"


class SimpleActiveLearning:

    def __init__(self, job_name, label_category_name,
                 label_names, max_selections):
        self.job_name = job_name
        self.label_category_name = label_category_name
        self.label_names = label_names
        self.max_selections = max_selections

    def compute_margin(self, probabilities, labels):
        """
       compute the confidence and the best label given the probability distribution.
      """
        max_probability = max(probabilities)
        max_prob_index = probabilities.index(max_probability)
        best_label = labels[max_prob_index]
        remaining_probs = [prob for i, prob in enumerate(probabilities) if i != max_prob_index]
        second_probability = max(remaining_probs, default=0.0)
        return max_probability - second_probability, best_label

    def get_label_index(self, inference_label_output):
        """
            inference_label_output is of the format "__label__0".
            This method gets an integer suffix from the end of the string.
            For this example, "__label__0" the function returns 0.
        """
        return int(inference_label_output.split('_')[-1])

    def make_metadata(self, margin, best_label):
        """
         make required metadata to match the output label.
      """
        return {
            'confidence': float(f'{margin: 1.2f}'),
            'job-name': self.job_name,
            'class-name': self.label_names[self.get_label_index(best_label)],
            'human-annotated': 'no',
            'creation-date': datetime.utcnow().strftime('%Y-%m-%dT%H:%m:%S.%f'),
            'type': JOB_TYPE
        }

    def make_autoannotation(self, prediction, source, margin, best_label):
        """
         generate the final output prediction with the label and confidence.
      """
        return {
            'source': source['source'],
            'id': prediction['id'],
            f'{self.label_category_name}': best_label,
            f'{self.label_category_name}-metadata': self.make_metadata(margin,
                                                                       best_label)
        }

    def autoannotate(self, predictions, sources):
        """
         auto annotate all unlabeled data with confidence above AUTOANNOTATION_THRESHOLD.
       """
        sources_by_id = {
            source['id']: source for source in sources
        }
        autoannotations = []
        for prediction in predictions:
            probabilities = prediction['prob']
            labels = prediction['label']
            margin, best_label = self.compute_margin(probabilities, labels)
            if margin > AUTOANNOTATION_THRESHOLD:
                autoannotations.append(self.make_autoannotation(
                    prediction, sources_by_id[prediction['id']],
                    margin, best_label
                ))

        return autoannotations

    def select_for_labeling(self, predictions, autoannotations):
        """
         Select the next set of records to be labeled by humans.
       """
        initial_ids = {
            prediction['id'] for prediction in predictions
        }
        autoannotation_ids = {
            autoannotation['id'] for autoannotation in autoannotations
        }
        remaining_ids = initial_ids - autoannotation_ids
        selections = random.sample(
            remaining_ids, min(self.max_selections, len(remaining_ids))
        )
        return selections


class ImageActiveLearning(SimpleActiveLearning):

    def __init__(self, job_name, label_category_name,
                 class_map, max_selections):
        self.job_name = job_name
        self.label_category_name = label_category_name
        self.class_map = class_map
        self.max_selections = max_selections

    def make_metadata(self, annotations):
        """
        Make required metadata to match the output label.
        """
        confidences = [a["score"] for a in annotations]
        objects = [{"confidence": c} for c in confidences]  # list of dicts e.g. [{"confidence": 0.5}]
        return {
            'objects': objects,
            'job-name': self.job_name,
            'class-map': self.class_map,
            'human-annotated': 'no',
            'creation-date': datetime.utcnow().strftime('%Y-%m-%dT%H:%m:%S.%f'),
            'type': JOB_TYPE
        }

    def make_autoannotation(self, prediction, source, annotations):
        """
        Generate the final output prediction with the label and confidence.
        """
        source_ref = source['source-ref']
        # get image dimensions by downloading image data
        image_bytesio = download_bytesio(S3Ref.from_uri(source_ref))
        image = np.array(Image.open(image_bytesio))
        image_height, image_width, depth = image.shape

        # annotations are 0-1 normalized, so the numbers should be multiplied by image dimensions
        for annotation in annotations:
            annotation['top'] = int(annotation['top'] * image_height)
            annotation['left'] = int(annotation['left'] * image_width)
            annotation['height'] = int(annotation['height'] * image_height)
            annotation['width'] = int(annotation['width'] * image_width)

        autoannotation_row = {
            'source-ref': source_ref,
            'id': source['id'],
            self.label_category_name: {
                'annotations': annotations,  # list of dicts
                'image_size': {
                    "width": image_width,
                    "height": image_height,
                    "depth": depth
                }
            },
            '{}-metadata'.format(self.label_category_name): self.make_metadata(annotations)
        }
        return autoannotation_row

    def autoannotate(self, predictions, sources):
        """
        Given the aligned lines of manifest file and inference output,
        auto annotate all unlabeled data with confidence above AUTOANNOTATION_THRESHOLD.
        Assume the default object detection response structure, where the prediction[1] is the confidence score.
        """
        logging.info("Autoannotating based on confidence threshold {:.2f}".format(AUTOANNOTATION_THRESHOLD))

        autoannotations = []
        for source, prediction in zip(sources, predictions):
            # if any of the detection results has low confidence, there may be false positive.
            # by default, false positive should be returned to the human annotator
            annotations = []  # follow the SageMaker bounding box manifest format
            make_autoannotation_yes = True
            for pred in prediction['prediction']:
                class_id, confidence_score, xmin, ymin, xmax, ymax = pred
                margin = confidence_score
                if margin < AUTOANNOTATION_THRESHOLD:
                    # any false positive should render the prediction invalid
                    make_autoannotation_yes = False
                    break
                top = ymin
                left = xmin
                width = xmax - xmin
                height = ymax - ymin
                annotation = {
                    'class_id': class_id, 'top': top, 'left': left, 'width': width, 'height': height, 'score': margin
                }
                annotations.append(annotation)

            if make_autoannotation_yes:
                # predictions are good enough, so make autoannotations
                autoannotations.append(self.make_autoannotation(prediction, source, annotations))
        logging.info("Populated autoannotation entries for {:d} samples".format(len(autoannotations)))
        return autoannotations

    def select_for_labeling(self, sources, autoannotations):
        """
        Select the next set of records to be labeled by humans.
        Do this by looking at which predictions ended up without associated auto-annotations.
        """
        initial_ids = {
            prediction['id'] for prediction in sources
        }
        autoannotation_ids = {
            autoannotation['id'] for autoannotation in autoannotations
        }
        remaining_ids = initial_ids - autoannotation_ids
        selections = random.sample(
            remaining_ids, min(self.max_selections, len(remaining_ids))
        )
        logging.info("The following ids were selected for labeling: {:s}".format(str(selections)))
        return selections