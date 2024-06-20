# scott

create database scott default character set utf8 collate utf8_general_ci;
show databases;

use scott;

CREATE TABLE DEPT ( 
	DEPTNO INT(2) PRIMARY KEY,  -- 부서번호
	DNAME VARCHAR(14), -- 부서명
	LOC VARCHAR(13) ); -- 지역명 
    
CREATE TABLE EMP (
	EMPNO INT(4) PRIMARY KEY, -- 사원번호
	ENAME VARCHAR(10), -- 사원이름
	JOB VARCHAR(9), -- 업무명
	MGR INT(4), -- 상사(사번) -> EMPNO 자기참조
	HIREDATE DATE, -- 입사일
	SAL FLOAT(7,2), -- 급여 
	COMM FLOAT(7,2), -- 커미션(수당)
	DEPTNO INT(2) NOT NULL, -- 부서번호 
    CONSTRAINT FK_DEPTNO FOREIGN KEY(DEPTNO) REFERENCES DEPT(DEPTNO));
    
INSERT INTO DEPT VALUES	(10,'ACCOUNTING','NEW YORK');
INSERT INTO DEPT VALUES (20,'RESEARCH','DALLAS');
INSERT INTO DEPT VALUES	(30,'SALES','CHICAGO');
INSERT INTO DEPT VALUES	(40,'OPERATIONS','BOSTON');

-- clerk 사무원
-- MGR 상사  
-- str_to_date 입사일


INSERT INTO EMP VALUES (7369,'SMITH','CLERK',7902,str_to_date('17-12-1980','%d-%m-%Y'),800,NULL,20);
INSERT INTO EMP VALUES (7499,'ALLEN','SALESMAN',7698,str_to_date('20-2-1981','%d-%m-%Y'),1600,300,30);
INSERT INTO EMP VALUES (7521,'WARD','SALESMAN',7698,str_to_date('22-2-1981','%d-%m-%Y'),1250,500,30);
INSERT INTO EMP VALUES (7566,'JONES','MANAGER',7839,str_to_date('2-4-1981','%d-%m-%Y'),2975,NULL,20);
INSERT INTO EMP VALUES (7654,'MARTIN','SALESMAN',7698,str_to_date('28-9-1981','%d-%m-%Y'),1250,1400,30);
INSERT INTO EMP VALUES (7698,'BLAKE','MANAGER',7839,str_to_date('1-5-1981','%d-%m-%Y'),2850,NULL,30);
INSERT INTO EMP VALUES (7782,'CLARK','MANAGER',7839,str_to_date('9-6-1981','%d-%m-%Y'),2450,NULL,10);
INSERT INTO EMP VALUES (7788,'SCOTT','ANALYST',7566,str_to_date('13-7-87', '%d-%m-%Y'),3000,NULL,20);
INSERT INTO EMP VALUES (7839,'KING','PRESIDENT',NULL,str_to_date('17-11-1981','%d-%m-%Y'),5000,NULL,10);  -- president 상사없음. 
INSERT INTO EMP VALUES (7844,'TURNER','SALESMAN',7698,str_to_date('8-9-1981','%d-%m-%Y'),1500,0,30);
INSERT INTO EMP VALUES (7876,'ADAMS','CLERK',7788,str_to_date('13-7-87', '%d-%m-%Y'),1100,NULL,20);
INSERT INTO EMP VALUES (7900,'JAMES','CLERK',7698,str_to_date('3-12-1981','%d-%m-%Y'),950,NULL,30);
INSERT INTO EMP VALUES (7902,'FORD','ANALYST',7566,str_to_date('3-12-1981','%d-%m-%Y'),3000,NULL,20);
INSERT INTO EMP VALUES (7934,'MILLER','CLERK',7782,str_to_date('23-1-1982','%d-%m-%Y'),1300,NULL,10);
-- 보너스
CREATE TABLE BONUS (
	ENAME VARCHAR(10), -- 사원이름
	JOB VARCHAR(9), -- 업무명
	SAL INT, -- 급여 
	COMM INT); -- 커미션(수당)
-- 급여등급
CREATE TABLE SALGRADE ( 
	GRADE INT, -- 급여등급
	LOSAL INT, -- 급여 하한 값
	HISAL INT); -- 급여 상한 값
    
INSERT INTO SALGRADE VALUES (1,700,1200);  -- 1 -> 호봉 
INSERT INTO SALGRADE VALUES (2,1201,1400);
INSERT INTO SALGRADE VALUES (3,1401,2000);
INSERT INTO SALGRADE VALUES (4,2001,3000);
INSERT INTO SALGRADE VALUES (5,3001,9999);    

select * from dept;
select count(*) from dept; -- 4건

select * from emp;
select count(*) from emp; -- 14건

select * from SALGRADE;
select count(*) from SALGRADE; -- 5건

select * from BONUS;
select count(*) from BONUS;

