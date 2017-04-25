import os
import sys
path='datasets'

def Main(argv):
    try:
        data_name=argv[0]
        method=argv[1]
    except:
        print('python collate_stats.py data_name')

    print(data_name)
    path='datasets'
    for dirname in os.listdir(os.path.join(os.getcwd(),path,str(data_name))):
        if os.path.isfile( os.path.join(os.getcwd(),path,str(data_name),dirname)):
            continue
        for epoch in os.listdir(os.path.join(os.getcwd(),path,str(data_name),dirname)):
            if os.path.isfile( os.path.join(os.getcwd(),path,str(data_name),dirname,epoch)   ):
                continue
            print('python '+ method  +' ' +str(dirname)+' '+str(epoch)+' '+data_name)
            os.system('python '+ method +' '+str(dirname)+' '+str(epoch)+' '+data_name)

if __name__=='__main__':
    Main(sys.argv[1:])
