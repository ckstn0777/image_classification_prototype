from classifier import ImageClassifier


class BentoML:
    def run_bentoml(self, model):
        classifier_service = ImageClassifier()
        classifier_service.pack('classifier', model)

        saved_path = classifier_service.save()
        print("save path : ", saved_path)
