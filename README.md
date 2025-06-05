# Robotic Singulation and Grasping of Plant Seedlings for Autonomous Transplanting 

In the heart of agriculture, one persistent challenge continues to trouble modern farms — the shortage of labor. Among the many tasks in need of attention, one stands out for its delicacy and dexterity: transplanting vegetatively propagated crops such as sweetpotato which are grown using plant cuttings (i.e. slips). These are not seeds or large tubers, but slender, fragile cuttings that must be carefully separated, oriented, and planted — a process that has stubbornly remained manual due to its complexity.
But what if robots could lend a hand?
That’s the question we set out to explore. While automated machines starting to emerge that can transplant slips once they're sorted and arranged in chain pots, the real bottleneck lies upstream — in the slip sorting process. The challenge? These slips are not only thin and flexible but also tend to overlap, twist, and cling to one another.
At the core of our system is a 6-degree-of-freedom robotic arm, equipped with a parallel-jaw gripper and a 3D camera mounted in an eye-in-hand configuration. The brain of the system is a deep learning model that performs two key tasks: instance segmentation and keypoint detection. First, the model scans the messy pile of slips and identifies each one as a separate instance and determine transplanting direction. With this information in hand, the robot plans its next move. A grasp-planning algorithm calculates the best, collision-free way to pick up each slip, carefully extracting it from the clutter without disturbing the rest. It then places the slip into a chain pot to grow.
We tested this robotic framework in real-world experiments, and the results were promising. With over 80% success rate, our robot was able to automatically sort and transplant sweetpotato slips.
The future work would be to address the challenges faced in heavily cluttered or highly overlapping scenarios, where it occasionally performs multi-picks or selects the wrong transplanting direction.

<p align="center">
  <img src="https://github.com/NCSU-BAE-ARLab/Sweetpotato_transplanting/blob/main/assets/Github1.gif" width="49%" />
  <img src="https://github.com/NCSU-BAE-ARLab/Sweetpotato_transplanting/blob/main/assets/github_vrop2.gif" width="49%" />
</p>

<p align="center">
  <img src="https://github.com/NCSU-BAE-ARLab/Sweetpotato_transplanting/blob/main/assets/grasping_framework.png" />
</p>
