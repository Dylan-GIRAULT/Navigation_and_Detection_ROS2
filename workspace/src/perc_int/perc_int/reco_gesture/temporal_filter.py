from collections import deque, Counter


class TemporalFilter:
    def __init__(self, window=7, debounce=4):
        self.window = window
        self.debounce = debounce
        self.labels = deque(maxlen=window)
        self.confidences = deque(maxlen=window)
        self.last_output = None
        self.last_count = 0

    def add(self, label, confidence):
        self.labels.append(label)
        self.confidences.append(confidence)

        cnt = Counter(self.labels)
        label_most, count = cnt.most_common(1)[0]
        if label_most != self.last_output:

            if count >= self.debounce:
                self.last_output = label_most
                self.last_count = count
        return self.last_output, (sum(self.confidences) / len(self.confidences) if self.confidences else 0.0)