package com.cloudera.sparkts.api.java;

public class TestBreak {

    public static void main(String[] avgs) {
        L100 : for(int j = 0; j < 5; j++) L200 : for (int i = 0; i<5; i++) {
                if (i == 3) {
                    break L100;
                }
                System.out.println(i);
            }
        System.out.println("End");
    }
}
