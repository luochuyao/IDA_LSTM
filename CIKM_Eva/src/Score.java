public class Score {

    private double[] dBZ_thresholds = {5.0, 20.0 ,40.0};
    private double[][] ret = new double[this.dBZ_thresholds.length][4];
    private double[] pixel_thresholds = new double[this.dBZ_thresholds.length];
    private double[] hss = new double[this.dBZ_thresholds.length];
    private double[] pod = new double[this.dBZ_thresholds.length];
    private double[] far = new double[this.dBZ_thresholds.length];
    private double[] csi = new double[this.dBZ_thresholds.length];
    private double[] a = new double[this.dBZ_thresholds.length];
    private double[] b = new double[this.dBZ_thresholds.length];
    private double[] c = new double[this.dBZ_thresholds.length];
    private double[] d = new double[this.dBZ_thresholds.length];

    private int thresholds_num;

    public Score(){
        this.pixel_thresholds = this.dBZ_to_pixel(this.dBZ_thresholds);
        this.thresholds_num = this.dBZ_thresholds.length;
    }
    private double[] dBZ_to_pixel(double[] dBZ){
        double a = 58.53;
        double b = 1.56;
        double[] pixel_thresholds = new double[dBZ.length];

        for(int i = 0; i<pixel_thresholds.length;i++){
            pixel_thresholds[i] = (10+dBZ[i])*255.0/95.0;
        }
        return pixel_thresholds;

    }
    public double[] getCsi() {
        return this.csi;
    }
    public double[] getHss() {
        return this.hss;
    }
    public double[] getPod() { return this.pod; }
    public double[] getFar() { return this.far; }

    public boolean judeg_NaN(double[] data){
        for(int i = 0;i<data.length;i++){
            if (Double.isNaN(data[i])) {
                return true;
            }
        }
        return false;
    }

    public void printHss(){
        for(int i =0;i<this.hss.length;i++){
            System.out.println(this.hss[i]);
        }
        System.out.println("the averge of HSS is:"+this.get_average(this.hss));
    }
    public void printCsi(){
        for(int i =0;i<this.csi.length;i++){
            System.out.println(this.csi[i]);
        }
        System.out.println("the averge of CSI is:"+this.get_average(this.csi));
    }

    public void printPod(){
        for(int i =0;i<this.pod.length;i++){
            System.out.println(this.pod[i]);
        }
        System.out.println("the averge of POD is:"+this.get_average(this.pod));
    }
    public void printFar(){
        for(int i =0;i<this.far.length;i++){
            System.out.println(this.far[i]);
        }
        System.out.println("the averge of FAR is:"+this.get_average(this.far));
    }
    public double get_average(double[] data){
        double sum = 0.0;
        for(int i =0;i<data.length;i++){
            sum = sum+data[i];
        }
        return sum / data.length;

    }
    public double[] average(double[] data, int n){
        double[] result = new double[data.length];
        for(int i=0;i<data.length;i++){
            result[i] = data[i]/n;
        }
        return result;
    }

    private double[] get_column(int column_index,double[][] data) {
        int rows_num = data.length;//行数
//        int columns_num = data[0].length;//列数
        double[] new_data = new double[rows_num];
        for(int i=0;i<rows_num;i++){
            new_data[i] = data[i][column_index];
        }

        return new_data;
    }

    public double[] additive(double[] matrix_a, double[] matrix_b){
        double[] result = new double[matrix_a.length];
        for(int i=0;i<matrix_a.length;i++){
            result[i] = matrix_a[i]+matrix_b[i];
        }
        return result;
    }

    public double[] subtraction(double[] matrix_a, double[] matrix_b){
        double[] result = new double[matrix_a.length];
        for(int i=0;i<matrix_a.length;i++){
            result[i] = matrix_a[i]-matrix_b[i];
        }
        return result;
    }

    public double[] multiplication(double[] matrix_a, double[] matrix_b){
        double[] result = new double[matrix_a.length];
        for(int i=0;i<matrix_a.length;i++){
            result[i] = matrix_a[i] * matrix_b[i];
        }
        return result;
    }

    public double[] divide(double[] matrix_a, double[] matrix_b){
        double[] result = new double[matrix_a.length];
        for(int i=0;i<matrix_a.length;i++){
            result[i] = matrix_a[i] / matrix_b[i];
        }
        return result;
    }
    public double[] severalfold(int n, double[] matrix){
        double[] result = new double[matrix.length];
        for(int i=0;i<matrix.length;i++){
            result[i] = n*matrix[i];
        }
        return result;
    }
    public double[] add_constant(int n, double[] matrix){
        double[] result = new double[matrix.length];
        for(int i=0;i<matrix.length;i++){
            result[i] = n+matrix[i];
        }
        return result;
    }


    public void calculate_ret(int[] pred_img, int[] real_img) {
//        double[][] ret = new double[this.thresholds_num][4];
        for(int i=0;i<real_img.length;i++){
            for(int k = 0; k < this.thresholds_num ;k++){
                int bpred = (pred_img[i] > this.pixel_thresholds[k])? 1:0;
                int btruth = (real_img[i] > this.pixel_thresholds[k])? 1:0;
                int ind = (1 - btruth) * 2 + (1 - bpred);
                this.ret[k][ind] += 1;
            }
        }
//        double[] a = this.get_column(0,this.ret);
//        double[] c = this.get_column(1,this.ret);
//        double[] b = this.get_column(2,this.ret);
//        double[] d = this.get_column(3,this.ret);
//        this.a = this.additive(this.a,a);
//        this.b = this.additive(this.b,b);
//        this.c = this.additive(this.c,c);
//        this.d = this.additive(this.d,d);

    }

    public void count_ret(){
//        double[] a = this.a;
//        double[] c = this.c;
//        double[] b = this.b;
//        double[] d = this.d;
        double[] a = this.get_column(0,this.ret);
        double[] c = this.get_column(1,this.ret);
        double[] b = this.get_column(2,this.ret);
        double[] d = this.get_column(3,this.ret);

        this.csi = this.divide(a,this.additive(a,this.additive(b,c)));
        this.pod = this.divide(a,this.additive(a,c));
        this.far = this.divide(b,this.additive(a,b));
        double[] n = this.additive(a,this.additive(b,this.additive(c,d)));
        double[] aref = this.multiplication(this.divide(this.additive(a,b),n),this.additive(a,c));
        double[] gss = this.divide(this.subtraction(a,aref),this.subtraction(this.additive(a,this.additive(b,c)),aref));
        this.hss = this.divide(this.severalfold(2,gss),this.add_constant(1,gss));

    }
}
