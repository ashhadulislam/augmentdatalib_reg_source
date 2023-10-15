from unittest import TestCase

from ..data_augment import DataAugment


class TestTrain(TestCase):
	def test_training_data(self):


		'''Creation of sample data of two classes
		"%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% '''

		redclass=[[1,2],[3,4],[2,5]]
		redclass=np.array(redclass)
		x1=redclass*2-1;
		redclass=np.concatenate((redclass,x1))
		print("redclass\n",redclass)
		Otherdata=np.concatenate((redclass*2-2,redclass*2-3))
		Otherdata=np.concatenate(( Otherdata, np.zeros((len(Otherdata),1)) ),axis=1)
		print("Otherdata\n",Otherdata)		
		redclass=np.concatenate(( redclass, np.ones((len(redclass),1)) ),axis=1)
		print("redclass\n",redclass)
		Data=np.concatenate((redclass,Otherdata))
		x=redclass

		## Calling the function
		k=2
		randmx=1
		[Data_a,Ext_d,Ext_not]=DataAug(Data,k,1,80000,randmx)		