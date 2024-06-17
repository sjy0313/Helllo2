package ch05_.references.sec05.strings;

public class CharAtExample2 {
    public static void main(String[] args) {
        String ssn = "안녕하세요.";
        
        // 문자열의 요소를 배열형태로 할 수 없다 
        // System.out.println(ssn[0]);
        
        char str = ssn.charAt(2); // 문자열의 세 번째 문자를 가져옴
        System.out.println(ssn.length() + ", " + str); // 문자열의 길이와 세 번째 문자 출력
        
        for (int n = 0; n < ssn.length(); n++) { // for 루프 구문 수정
            System.out.println(ssn.charAt(n)); // 문자열의 각 문자를 출력
        }
    }
}
