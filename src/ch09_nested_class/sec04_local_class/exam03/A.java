package ch09_nested_class.sec04_local_class.exam03;
/*
 * 로컬변수를 로컬클래스에서 사용할 경우 
 * - 로컬변수는 final 특성을 갖게 됨
 * - 로컬변수 수정 불가
 * Java 에서는 로컬 클래스가 메소드의 로컬 변수나 인자를 접근할 때, 
 * 해당 변수들이 암묵적으로 final 이어야 합니다. Java 8 이후로는 effectively final 이라는 개념이 도입되어, 
 * 로컬 클래스에서 참조되는 변수는 명시적으로 final 로 선언되지 않더라도 값이 수정되지 않으면 참조할 수 있습니다.
 * 그러나 이러한 변수들은 값이 변경될 수 없음.값이 변경되려는 시도가 있으면 컴파일 오류가 발생

 */
public class A {
    // 메소드
    public void method1(int arg) { // final int arg
        // 로컬 변수
        int var = 1; // final int var = 1;
        
        // 로컬 클래스
        class B {
            // 메소드
            void method2() { // method 가 실행할 때만 B 객체를 생성할 수 있음
                // 로컬 변수 읽기
                System.out.println("arg: " + arg);  // (o)
                System.out.println("var: " + var);  // (o)
                
                // 로컬 변수 수정 불가 : final 이기 때문에
                // arg = 2;  // (x)
                // var = 2;  // (x)
            }
        }
        
        // 로컬 객체 생성
        B b = new B();
        // 로컬 객체 메소드 호출
        b.method2();
        
        // 로컬 변수 수정
        // arg = 3;  // (x)
        // var = 3;  // (x)
    }
    
}
