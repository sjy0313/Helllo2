package ch07_.inheritance.sec06_protected.package1;
/*
 * protected :
 * - private 처럼 접근 제한을 하지만 자식에게는 공개
 * 제한대상 : 필드 / 생산자 / 메소드
 * 제한범위 : 같은 패키지 / 자식
 */
public class A {
    // 필드 선언
    protected String field;

    // 생성자 선언
    protected A() {
        System.out.println("A() 생성자");
    }

    // 메소드 선언
    protected void method() {
        System.out.println("A.field :" + this.field);
    }
}

