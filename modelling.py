

from cv2 import resize
import matplotlib.image as mpimg
from numpy import array
from pandas import DataFrame

from features import (
    contrast,
    lightness,
    luminance,
    supracontrast,
    supralightness,
    supraluminance
)

featurizers = {
    "contrast":contrast,
    "lightness":lightness,
    "luminance":luminance,
    "supracontrast":supracontrast,
    "supralightness":supralightness,
    "supraluminance":supraluminance
}

class Model:
    def __init__(
        self,
        classifier,
        trainset,
        testset,
        features,
        classes,
        image_size
    ):
        self.classifier = classifier
        self.trainset = trainset
        self.testset = testset
        self.features = features
        self.classes = classes
        self.traintest = False
        
    
    def train(self):
        features, labels = self.trainset
        self.classifier.fit(features, labels)
    
    def test(self):
        features, labels = self.testset
        predictions = self.classifier.predict(features)
        
        return testreport(labels, predictions)
    
    def predict(self, rgb_image):
        featurevector=[]
        image = resize(rgb_image, image_size)
        
        for feature in self.features:
            featurizer = featurizers.get(feature)
            featurevector.append(
                featurizer(rgb_image)
            )
        
        featurevector = array([featurevector])
        print(featurevector)
        
        prediction = self.classifier.predict(featurevector)
        prediction = self.classes[prediction[0]]
        
        return prediction
        

def testreport(
    actual_labels,
    predictions,
    classes_=(0, 1)
):
    exp_label_count = {x:0 for x in classes_}
    pred_label_count = {x:0 for x in classes_}
    tdatasize = 0
    truepossize = 0
    falsepos = []

    for i in range(len(predictions)):
        
        m, n = actual_labels[i], predictions[i]
        if m==n:
            pred_label_count[m] += 1
            truepossize += 1
        else:
            falsepos.append(i)
        
        tdatasize += 1
        exp_label_count[m] += 1

    preds = []
    for key in pred_label_count:
        p_count = pred_label_count[key]
        e_count = exp_label_count[key]

        preds.append(
            "Class {}: {}/{} --> {}%".format(
                key,
                p_count, 
                e_count,
                round((p_count/e_count)*100, 2)
            )
        )
    
    accuracy = round((truepossize/tdatasize) * 100, 2)
    preds = "".join([
        "Overall Accuracy -> {}%\n\n".format(accuracy),
        "Class Accuracies:\n",
        "\n".join(preds)
    ])
    
    return preds, tuple(falsepos)
