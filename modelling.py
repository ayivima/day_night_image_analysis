"""Model Class and Test Report Generation Function"""

__author__ = "Victor Mawusi Ayi"

import matplotlib.image as mpimg
from cv2 import resize
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
        classes,
        imagesize,
        features=list(featurizers.keys()),
        accuracy=0,
        istrained=False
    ):
        self.classifier = classifier
        self.features = features
        self.classes = classes
        self.imagesize=imagesize
        self.accuracy = 0
        self.istrained = istrained

    def train(self, trainset):
        features, labels = trainset
        self.classifier.fit(features[self.features], labels)
        self.istrained = True
    
    def test(self, testset):
        
        if self.istrained == False:
            raise ValueError("Model is not trained yet")

        features, labels = testset
        predictions = self.classifier.predict(
            features[self.features]
        )
        
        report = testreport(labels, predictions, (0, 1))
        self.accuracy = report[0]
        
        return report
    
    def predict(self, rgb_image):
        
        if self.istrained == False:
            raise ValueError("Model is not trained yet")

        featurevector=[]
        image = resize(rgb_image, self.imagesize)
        
        for feature in self.features:
            featurizer = featurizers.get(feature)
            featurevector.append(
                featurizer(rgb_image)
            )
        
        prediction = self.classifier.predict([featurevector])
        prediction = self.classes[prediction[0]]
        
        return prediction



def testreport(
    actual_labels,
    predictions,
    classes_
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
