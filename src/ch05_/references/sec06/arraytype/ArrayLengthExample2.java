package ch05_.references.sec06.arraytype;
//https://steemit.com/kr/@modolee/floating-point - 부동소수점
public class ArrayLengthExample2 {
    public static void main(String[] args) {
        // 배열 변수 선언 및 초기화
        int[] scores = new int[] { 83, 90, 87 };

        // 총합 및 평균 계산 후 출력
        System.out.println("총합: " + totalScore(scores));
        System.out.println("평균: " + averageScore(scores));
    }
    
    // 총합
    public static int totalScore(int[] scores) {
        int total = 0; // 총합 초기화
        for(int i = 0; i < scores.length; i++) {
            total += scores[i]; // 배열 요소를 더하여 총합 계산
        }
        return total; // 총합 반환
    }
    
    // 평균
    public static double averageScore(int[] scores) {
        int total = totalScore(scores);
        double average = (double) total / scores.length;
        return average; 
    }
}
