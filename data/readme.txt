Here are some explanations about the files:

1) dialogues_text.txt: The DailyDialog dataset which contains 11318 transcribed dialogues.
2) dialogues_topic.txt: Each line of dialogues_topic.txt corresponds to the topic of same line in dialogues_text.txt.
                        The representation of the topic number: {1: Ordinary Life, 2: School Life, 3: Culture & Education,
                        4: Attitude & Emotion, 5: Relationship, 6: Tourism , 7: Health, 8: Work, 9: Politics, 10: Finance}
3) dialogues_act.txt: Each line of dialogues_act.txt corresponds to the act of same line in dialogues_text.txt.
                      The representation of the act number: { 1: inform，2: question, 3: directive, 4: commissive }
4) dialogues_emotion.txt: Each line of dialogues_emotion.txt corresponds to the emotion of same line in dialogues_text.txt.
                          The representation of the emotion number: { 0: no emotion, 1: anger, 2: disgust, 3: fear, 4: happiness, 5: sadness, 6: surprise}
5) train.zip, validation.zip and test.zip are the different segmentation of the whole dataset. 
