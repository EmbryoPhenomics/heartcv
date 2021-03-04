import pytest
import heartcv as hcv


def data():
	return [list(range(*args)) for args in ((0,10), (1,11), (20,40,2))]


def reverse(vals, ret):
	return ret * (max(vals) - min(vals)) + min(vals)


class Test_MinMaxScale:
	@pytest.mark.parametrize('data', data())
	def test_limits(self, data):
		ret = hcv.minmax_scale(data)

		assert min(ret) == 0
		assert max(ret) == 1

	@pytest.mark.parametrize('data', data())
	def test_reverse(self, data):
		ret = hcv.minmax_scale(data)
		
		assert data == reverse(data, ret).tolist()	

# class Test_MSE:
# 	def test