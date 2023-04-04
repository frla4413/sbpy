def export_to_tecplot(x,y,z,filename):

    filename = filename +'.tec'
    with open(filename,'w') as f:
        f.write("TITLE = 'solution.tec'\n")
        f.write('VARIABLES = x,y,u\n')

        f.write('ZONE I = ' + str(x.shape[0])+ \
                ', J = ' + str(y.shape[1])+ ', F = POINT\n')

        for j in range(x.shape[1]):
            for i in range(y.shape[0]):
                my_str = '%.3f %.3f %.3f\n'%(x[i,j],y[i,j],z[i,j])
                f.write(my_str)
        f.close()
