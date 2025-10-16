Title: Voice-Controlled Wheelchair

Objective: The primary goal of this practical assignment is to have students apply core
concepts of a data analysis pipeline. This includes the stages of data
preparation, cleaning, extraction of descriptive features, feature selection or
dimensionality reduction, and machine learning.

Problem: The problem proposed in this practical assignment is a typical requirements analysis
challenge often faced by Data Scientists. The exercise is set within the context of voice
command recognition for controlling a wheelchair. This scenario is highly significant, as
voice control of an electric wheelchair is often the only means of independent movement
for people with quadriplegia. The exercise will allow students to practice and internalize
key concepts of any data analysis pipeline that a data scientist encounters: given a large
volume of real data, students will analyze and identify a set of features that enable the
recognition of distinct states

Metodology:

In this assignment, we will consider five possible control commands:
• Forward: Move the wheelchair forward
• Backward: Move the wheelchair backward,
• Left: Turn the wheelchair to the left,
• Right: Turn the wheelchair to the right,
• Stop: Stop the wheelchair.

In addition to the five main commands, it is crucial for the system to recognize when no
command is being given. Since the microphone in such a system records continuously, we
will define two additional states that do not correspond to any command:
• Silence: Absence of sound, only background noise;
• Unknown: Miscellaneous words that are not commands and should therefore be
ignored.
The dataset is organized with a folder for each command (including a folder for silence
examples and another for unknown cases). Inside each folder is a set of files in ".wav"
format. Each file corresponds to a recording of the verbal command associated with that
folder.