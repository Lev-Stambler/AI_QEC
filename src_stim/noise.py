class NoisePauliChannel2:
	"""
	Representing the noise
	"""
	def __init__(self) -> None:
		pass

class NoisePauli:
	def __init__(self, p_x: float, p_y: float, p_z: float) -> None:
		self.p_x = p_x
		self.p_y = p_y
		self.p_z = p_z