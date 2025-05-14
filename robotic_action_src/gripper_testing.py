#!/usr/bin/env python
# license removed for brevity
# import rospy
# from std_msgs.msg import UInt16
import sys
import socket
import time
import csv
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

					s.sendall(b'SET FOR 0\n') 
					data_force = s.recv(1024).decode().strip()
					print(data_force)

					s.sendall(b'SET SPE 0\n') 
					data_force = s.recv(1024).decode().strip()
					print(data_force)

					s.sendall('SET POS {0}\n'.format(value).encode())
					data = s.recv(2**10)
			print('[Gripper]: moving to {0}'.format(value))
			
		else:
			print('attention: gripper is not being used for testing purposes')

	def print_current_values(self, filename="gripper_current_log.csv", duration=10000):
		with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s, open(filename, mode='w', newline='') as file:
			writer = csv.writer(file)
			writer.writerow(["Timestamp_ms", "Raw_COUs", "Current_mA"])

			s.connect((self.HOST, self.PORT))
			start_time = time.time()  # Record start time

			# s.sendall(b'SET FOR 10\n') 
			# data_force = s.recv(1024).decode().strip()
			# print(data_force)



			for i in range(duration):
				s.sendall(b'GET COU\n')
				data_cur = s.recv(1024).decode().strip()

				# s.sendall(b'SET FOR 1\n') 

				# s.sendall(b'GET FOR 1\n') 
				s.sendall(b'GET FOR\n') 
				data_force = s.recv(1024).decode().strip()
				print(data_force)

				s.sendall(b'GET SPE\n') 
				data_force = s.recv(1024).decode().strip()
				print(data_force)

				current_time = time.time()
				timestamp_ms = int((current_time - start_time) * 1000)  # milliseconds since start

				if data_cur.startswith("COU"):
					try:
						value_cur = int(data_cur.split()[1])
						current_mA = value_cur * 10
						writer.writerow([timestamp_ms, value_cur, current_mA])
						print(f"{i+1:03d}: {timestamp_ms} ms - Raw: {value_cur}, Current: {current_mA} mA")
						# if int(data_force.split()[1]) > 0:
						# 	print(int(data_force.split()[1]))
					except (IndexError, ValueError):
						print("Error parsing:", data)
				else:
					print("Unexpected response:", data)

				time.sleep(0.01)  # Adjust as needed



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
	gripper.print_current_values()
