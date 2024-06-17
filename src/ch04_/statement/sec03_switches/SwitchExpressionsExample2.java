package ch04_.statement.sec03_switches;
/*
 * [switch expressions]
 * java12 이후부터 지원 
 * which allows multiple case labels to be grouped together using a comma,and an arrow -> 
 * to indicate the action for each case.
 * break문을 없앰
 */
public class SwitchExpressionsExample2 {
    public static void main(String[] args) {
        char grade = 'a';
        // switch statement 
        switch (grade) {
            case 'A', 'a' -> {
                System.out.println("우수 회원입니다.");
            }
            case 'B', 'b' -> {
                System.out.println("일반 회원입니다.");
            }                        
            default -> {
                System.out.println("손님입니다.");
            }
        }
        // if-else statement
        if (grade == 'A' || grade == 'a') {
            System.out.println("우수 회원입니다.");
        } else if (grade == 'B' || grade == 'b') {
            System.out.println("일반 회원입니다.");
        } else {
            System.out.println("손님입니다.");
        }
    }
}
