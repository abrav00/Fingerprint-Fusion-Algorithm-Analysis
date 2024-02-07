from PIL import Image, ImageDraw
import argparse
import math

def get_hough_image(im):
    (x, y) = im.size
    x *= 1.0
    y *= 1.0

    im_load = im.load()

    result = Image.new("RGBA", im.size, 0)
    draw = ImageDraw.Draw(result)

    for i in range(im.size[0]):
        for j in range(im.size[1]):
            if im_load[i, j] > 220:
                line = lambda t: (t, (-(i / x - 0.5) * (t / x) + (j / y - 0.5)) * x)
                draw.line([line(0), line(x)], fill=(50, 0, 0, 10))

    return result

def extract_line_features(hough_image, threshold=220):
    im_load = hough_image.load()
    lines = []
    for i in range(hough_image.size[0]):
        start = None
        for j in range(hough_image.size[1]):
            if im_load[i, j] >= threshold:
                if start is None:
                    start = (i, j)
            else:
                if start is not None:
                    end = (i, j - 1)
                    lines.append((start, end))
                    start = None
        if start is not None:
            lines.append((start, (i, hough_image.size[1] - 1)))
    return lines

def calculate_feature_distance(feature1, feature2):
    (x1_start, y1_start), (x1_end, y1_end) = feature1
    (x2_start, y2_start), (x2_end, y2_end) = feature2

    distance_start = math.sqrt((x1_start - x2_start) ** 2 + (y1_start - y2_start) ** 2)
    distance_end = math.sqrt((x1_end - x2_end) ** 2 + (y1_end - y2_end) ** 2)

    return (distance_start + distance_end) / 2

def match_features(features1, features2):
    matches = []
    SOME_THRESHOLD = 5  # Example threshold
    for f1 in features1:
        best_match = None
        min_distance = float('inf')
        for f2 in features2:
            distance = calculate_feature_distance(f1, f2)
            if distance < min_distance:
                min_distance = distance
                best_match = f2
        if min_distance < SOME_THRESHOLD:
            matches.append((f1, best_match))
    return matches

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hough transform")
    parser.add_argument("image", nargs=1, help="Path to image")
    args = parser.parse_args()

    im = Image.open(args.image[0])
    im = im.convert("L")  # Convert to grayscale
    im.show()

    hough_img = get_hough_image(im)
    hough_img.show()

    features = extract_line_features(hough_img)
    matched_features = match_features(features, features)


