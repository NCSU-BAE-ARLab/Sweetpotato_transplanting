#!/usr/bin/env python
# license removed for brevity
# import rospy
# from std_msgs.msg import UInt16
import sys
import socket
import time
# import rosservice

class Gripper:
	def __init__(self,gripper_on):
		# rospy.init_node('gripper')
		self.gripper_on = gripper_on
		self.HOST="192.168.1.10" #replace by the IP address of the UR robot
		self.PORT=63352 #PORT used by robotiq gripper

		if gripper_on:
			with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
				#open the socket

				s.connect((self.HOST, self.PORT))

				s.sendall(b'GET ACT\n')
				data = s.recv(2**10)
				# print(data)
				if data == b'ACT 1\n':
					print('[Gripper]: active')
				else:
					s.sendall(b'SET ACT 1\n')
					print('Gripper activating...')
					time.sleep(8)


				s.sendall(b'GET GTO\n')
				data = s.recv(2**10)
				# print(data)
				if data != b'GTO 1\n':
					s.sendall(b'SET GTO 1\n')

			print('[Gripper]: ready to take the commands')
		

	def homing(self):
		with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
			s.connect((self.HOST, self.PORT))
			s.sendall(b'SET POS 0\n')
			data = s.recv(2**10)
		print('[Gripper] homing', data)


	def run(self,value):
		if self.gripper_on:
			if value > 255 or value < 0:
				print('Gripper  opening value must be between [0,255]. Aborting.')
				sys.exit(1)
			else:
				# print('Gripper should move')
				with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
					s.connect((self.HOST, self.PORT))
					# print('SET POS {0}\n'.format(value).encode())
					s.sendall('SET POS {0}\n'.format(value).encode())
					data = s.recv(2**10)
			print('[Gripper]: moving to {0}'.format(value))
			
		else:
			print('attention: gripper is not being used for testing purposes')

if __name__ == "__main__":
	if len(sys.argv) < 2:
		print('please provide gripper opening value')
		sys.exit()
	value = int(sys.argv[1])
	# print(value)
	# rospy.init_node('gripper')
	gripper = Gripper(gripper_on=True)
	# gripper.homing()
	gripper.run(value)