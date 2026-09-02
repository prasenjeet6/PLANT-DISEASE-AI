def check_reliability(result):

    mobile_prediction = result["mobilenet_prediction"]
    efficient_prediction = result["efficientnet_prediction"]

    mobile_confidence = result["mobilenet_confidence"]
    efficient_confidence = result["efficientnet_confidence"]

    if mobile_prediction == efficient_prediction:

        agreement = True

        average_confidence = (
            mobile_confidence + efficient_confidence
        ) / 2

        if average_confidence >= 95:
            reliability = "Very High"

        elif average_confidence >= 85:
            reliability = "High"

        elif average_confidence >= 70:
            reliability = "Medium"

        else:
            reliability = "Low"

    else:

        agreement = False

        average_confidence = (
            mobile_confidence + efficient_confidence
        ) / 2

        reliability = "Low"

    return {
        "agreement": agreement,
        "reliability": reliability,
        "average_confidence": average_confidence
    }