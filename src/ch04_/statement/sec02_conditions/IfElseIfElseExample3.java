package ch04_.statement.sec02_conditions;
/*
 * 점수는 0점부터 100까지 범위를 넘어서면 오류처리 
 * 점수가 잘못되었다는 출력처리 
 */
public class IfElseIfElseExample3 {
    public static void main(String[] args) {
        int score = -5;
        
        if(score >= 0 && score <= 100) { // ||도 가능
            if(score < 70) {
                System.out.println("점수가 70 미만입니다.");
                System.out.println("등급은 D입니다.");
            } else if(score < 80) { // python에서는 elif
                System.out.println("점수가 70~79입니다.");
                System.out.println("등급은 C입니다.");
            } else if(score < 90) {
                System.out.println("점수가 80~89입니다.");
                System.out.println("등급은 B입니다.");
            } else { // 90보다 크면
                System.out.println("점수가 90~100입니다.");
                System.out.println("등급은 A입니다.");
            }
        } else {
            System.out.printf("점수(%d)가 잘못되었습니다.%n", score); // printf 실수도 들어갈 수 있음.
        }
    }
}

//작은 값 ->큰 값