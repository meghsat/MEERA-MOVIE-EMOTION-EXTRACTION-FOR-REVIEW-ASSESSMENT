import cv2
import numpy as np
import time
import argparse
from inference_engine import MovieEmotionAnalyzer

def main():
    args = parser.parse_args()
    
    analyzer = MovieEmotionAnalyzer(args.model_dir)
    
    if args.input.isdigit():
        cap = cv2.VideoCapture(int(args.input))
    else:
        cap = cv2.VideoCapture(args.input)
    
    if not cap.isOpened():
        print(f"Error: Could not open video source {args.input}")
        return
    
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(args.output, fourcc, fps, (frame_width, frame_height))
    
    emotion_counts = {emotion: 0 for emotion in analyzer.emotion_labels}
    frame_count = 0
    total_processing_time = 0
    
    print("Starting emotion analysis...")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        result = analyzer.process_frame(frame)
        enhanced_frame = result["enhanced_frame"]
        face_results = result["face_results"]
        processing_time = result["processing_time"]
        
        total_processing_time += processing_time
        frame_count += 1
        
        for face_data in face_results:
            x, y, w, h = face_data["face_location"]
            emotion = face_data["emotion"]
            conf = face_data["emotion_confidence"]
            age = face_data["age_group"]
            gender = face_data["gender"]
            
            emotion_counts[emotion] += 1
            cv2.rectangle(enhanced_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            text = f"{emotion} ({conf:.2f}) | {gender}, {age}"
            cv2.putText(enhanced_frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        fps_text = f"FPS: {1.0/processing_time:.2f}" if processing_time > 0 else "FPS: N/A"
        cv2.putText(enhanced_frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        out.write(enhanced_frame)
        
        if args.display:
            cv2.imshow('Movie Theater Emotion Analysis', enhanced_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    for emotion, count in emotion_counts.items():
        percentage = (count / sum(emotion_counts.values())) * 100 if sum(emotion_counts.values()) > 0 else 0
        print(f"{emotion}: {count} ({percentage:.2f}%)")

if __name__ == "__main__":
    main()